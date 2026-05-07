import os
import gc
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# Constants for data processing
VOCAB_SIZE = 120000
SUFFIX_VOCAB_SIZE = 15000
PREFIX_VOCAB_SIZE = 7500
AMBIGUITY_THRESHOLD = 0.97
MAX_VARIANTS = 50
WINDOW_SIZE = 6
SEQ_LEN = 2 * WINDOW_SIZE + 1
MAX_CHAR_LEN = 20


def get_case_id(token):
    """Get casing category for a token."""
    if token.islower():
        return 1
    elif token.istitle():
        return 2
    elif token.isupper():
        return 3
    else:
        return 4


def add_context_features(df, window):
    """Add left and right context columns for each feature."""
    df = df.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)
    feat_cols = ['w_idx', 's_idx', 'p_idx', 'c_idx', 'u_idx']
    dtypes = [np.int32, np.int16, np.int16, np.int8, np.int32]
    for offset in range(1, window + 1):
        for col, dtype in zip(feat_cols, dtypes):
            # Left
            left = df.groupby('sentence_id')[col].shift(offset).fillna(0).astype(dtype)
            df[f'L{offset}_{col}'] = left
            # Right
            right = df.groupby('sentence_id')[col].shift(-offset).fillna(0).astype(dtype)
            df[f'R{offset}_{col}'] = right
    # Rename center columns
    rename_dict = {col: f'C_{col}' for col in feat_cols}
    df.rename(columns=rename_dict, inplace=True)
    return df


class MyDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = VOCAB_SIZE
        self.suffix_vocab_size = SUFFIX_VOCAB_SIZE
        self.prefix_vocab_size = PREFIX_VOCAB_SIZE
        self.max_variants = MAX_VARIANTS
        self.window_size = WINDOW_SIZE
        self.max_char_len = MAX_CHAR_LEN

    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        print('Loading data...')
        
        # Load training data
        train = pd.read_csv(
            './input/ru_train.csv',
            dtype={
                'sentence_id': 'int32',
                'token_id': 'int16',
                'class': 'category',
                'before': 'object',
                'after': 'object'
            },
            keep_default_na=False
        )

        # Load test data
        test = pd.read_csv(
            './input/ru_test_2.csv',
            dtype={
                'sentence_id': 'int32',
                'token_id': 'int16',
                'before': 'object'
            },
            keep_default_na=False
        )

        # Check for validation set - use fixed val.csv if exists
        val_sentence_ids = None
        if os.path.exists('./input/val.csv'):
            print('Loading validation set from val.csv...')
            val_df = pd.read_csv('./input/val.csv')
            val_sentence_ids = set(val_df['sentence_id'].unique())

        # Build token statistics
        print('Building token statistics...')
        token_counter = Counter(zip(train['before'], train['after']))
        token_stats = {}
        for (b, a), cnt in token_counter.items():
            token_stats.setdefault(b, []).append((a, cnt))

        for b in token_stats:
            token_stats[b].sort(key=lambda x: x[1], reverse=True)

        token_map = {}
        ambiguous_tokens = set()
        for b, lst in token_stats.items():
            total = sum(cnt for _, cnt in lst)
            top_cnt = lst[0][1]
            token_map[b] = [a for a, _ in lst[:MAX_VARIANTS]]
            if top_cnt / total < AMBIGUITY_THRESHOLD:
                ambiguous_tokens.add(b)

        # Build vocabularies
        print('Building vocabularies...')
        # Word vocabulary
        before_counts = train['before'].value_counts()
        top_before = before_counts.index[:VOCAB_SIZE].tolist()
        word2idx = {w: i + 1 for i, w in enumerate(top_before)}

        # Suffix vocabulary
        unique_train_before = train['before'].unique()
        suffix_counter = Counter()
        for token in unique_train_before:
            if len(token) >= 3:
                suffix_counter[token[-3:]] += 1
        top_suffixes = [suf for suf, _ in suffix_counter.most_common(SUFFIX_VOCAB_SIZE)]
        suffix2idx = {s: i + 1 for i, s in enumerate(top_suffixes)}

        # Prefix vocabulary
        prefix_counter = Counter()
        for token in unique_train_before:
            if len(token) >= 3:
                prefix_counter[token[:3]] += 1
        top_prefixes = [pre for pre, _ in prefix_counter.most_common(PREFIX_VOCAB_SIZE)]
        prefix2idx = {p: i + 1 for i, p in enumerate(top_prefixes)}

        # Character vocabulary
        all_tokens = set(train['before'].unique()) | set(test['before'].unique())
        all_chars = set()
        for token in all_tokens:
            all_chars.update(token)
        char_list = sorted(all_chars)
        char2idx = {ch: i + 1 for i, ch in enumerate(char_list)}

        # Unique token mapping
        all_unique_tokens = list(all_tokens)
        u_token2idx = {t: i + 1 for i, t in enumerate(all_unique_tokens)}

        # Character matrix
        print('Building character matrix...')
        num_u_tokens = len(all_unique_tokens)
        char_matrix = np.zeros((num_u_tokens + 1, MAX_CHAR_LEN), dtype=np.int16)
        for token, idx in tqdm(u_token2idx.items()):
            for j, ch in enumerate(token[:MAX_CHAR_LEN]):
                char_matrix[idx, j] = char2idx.get(ch, 0)

        # Feature mapping functions
        def map_word(token):
            return word2idx.get(token, 0)

        def map_suffix(token):
            if len(token) >= 3:
                return suffix2idx.get(token[-3:], 0)
            return 0

        def map_prefix(token):
            if len(token) >= 3:
                return prefix2idx.get(token[:3], 0)
            return 0

        def map_case(token):
            return get_case_id(token)

        def map_u(token):
            return u_token2idx.get(token, 0)

        # Map features on training data
        print('Mapping features...')
        train['w_idx'] = train['before'].map(map_word).astype('int32')
        train['s_idx'] = train['before'].map(map_suffix).astype('int16')
        train['p_idx'] = train['before'].map(map_prefix).astype('int16')
        train['c_idx'] = train['before'].map(map_case).astype('int8')
        train['u_idx'] = train['before'].map(map_u).astype('int32')

        # Map features on test data
        test['w_idx'] = test['before'].map(map_word).astype('int32')
        test['s_idx'] = test['before'].map(map_suffix).astype('int16')
        test['p_idx'] = test['before'].map(map_prefix).astype('int16')
        test['c_idx'] = test['before'].map(map_case).astype('int8')
        test['u_idx'] = test['before'].map(map_u).astype('int32')

        # Add context window features
        print('Adding context to training data...')
        train = add_context_features(train, WINDOW_SIZE)
        print('Adding context to test data...')
        test = add_context_features(test, WINDOW_SIZE)

        # Ambiguous flags and class encoding
        train['is_ambiguous'] = train['before'].isin(ambiguous_tokens)
        test['is_ambiguous'] = test['before'].isin(ambiguous_tokens)

        # Class encoding
        class_cats = train['class'].astype('category')
        train['class_code'] = class_cats.cat.codes.astype('int8')
        num_classes = len(class_cats.cat.categories)

        # Prepare ambiguous training data
        print('Preparing ambiguous training data...')
        ambig_df = train[train['is_ambiguous']].copy()

        # Target index within top candidates
        def get_target_idx(row):
            b = row['before']
            a = row['after']
            cand = token_map.get(b, [])
            try:
                return cand.index(a)
            except ValueError:
                return 0

        ambig_df['target'] = ambig_df.apply(get_target_idx, axis=1).astype('int8')

        # Extract sequence arrays
        pos_order = [f'L{i}' for i in range(WINDOW_SIZE, 0, -1)] + ['C'] + [f'R{i}' for i in range(1, WINDOW_SIZE + 1)]

        word_cols = [f'{pos}_w_idx' for pos in pos_order]
        suffix_cols = [f'{pos}_s_idx' for pos in pos_order]
        prefix_cols = [f'{pos}_p_idx' for pos in pos_order]
        case_cols = [f'{pos}_c_idx' for pos in pos_order]
        uidx_cols = [f'{pos}_u_idx' for pos in pos_order]

        X_word = ambig_df[word_cols].values.astype('int32')
        X_suffix = ambig_df[suffix_cols].values.astype('int16')
        X_prefix = ambig_df[prefix_cols].values.astype('int16')
        X_case = ambig_df[case_cols].values.astype('int8')
        X_uidx = ambig_df[uidx_cols].values.astype('int32')
        y_norm = ambig_df['target'].values.astype('int64')
        y_class = ambig_df['class_code'].values.astype('int64')
        sentence_ids = ambig_df['sentence_id'].values

        # Free memory
        del train, ambig_df
        gc.collect()

        # Prepare test data
        print('Preparing test data...')
        test_ambig = test[test['is_ambiguous']].copy()
        test_nonambig = test[~test['is_ambiguous']].copy()

        test_word_seq = test_ambig[word_cols].values.astype('int32')
        test_suffix_seq = test_ambig[suffix_cols].values.astype('int16')
        test_prefix_seq = test_ambig[prefix_cols].values.astype('int16')
        test_case_seq = test_ambig[case_cols].values.astype('int8')
        test_uidx_seq = test_ambig[uidx_cols].values.astype('int32')

        # Default after map for non-ambiguous tokens
        default_after_map = {b: lst[0] for b, lst in token_map.items()}

        # Store training data
        self.train_data = {
            'X_word': X_word,
            'X_suffix': X_suffix,
            'X_prefix': X_prefix,
            'X_case': X_case,
            'X_uidx': X_uidx,
            'y_norm': y_norm,
            'y_class': y_class,
            'sentence_ids': sentence_ids,
            'val_sentence_ids': val_sentence_ids,
            # Metadata
            'vocab_size': VOCAB_SIZE,
            'suffix_vocab_size': SUFFIX_VOCAB_SIZE,
            'prefix_vocab_size': PREFIX_VOCAB_SIZE,
            'max_variants': MAX_VARIANTS,
            'window_size': WINDOW_SIZE,
            'max_char_len': MAX_CHAR_LEN,
            'num_classes': num_classes,
            'char_vocab_size': len(char2idx),
            'char_matrix': char_matrix,
        }

        # Store test data
        self.test_data = {
            'test_ambig': test_ambig,
            'test_nonambig': test_nonambig,
            'test_word_seq': test_word_seq,
            'test_suffix_seq': test_suffix_seq,
            'test_prefix_seq': test_prefix_seq,
            'test_case_seq': test_case_seq,
            'test_uidx_seq': test_uidx_seq,
            'word_cols': word_cols,
            'token_map': token_map,
            'default_after_map': default_after_map,
        }

    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return """
        Data processing for Russian text normalization:
        - Loads training and test data from CSV files
        - Builds vocabularies for words (top 120k), suffixes (top 15k), prefixes (top 7.5k), and characters
        - Creates character-level feature matrix for all unique tokens
        - Adds context window features (6 tokens on each side, total 13 positions)
        - Identifies ambiguous tokens based on frequency threshold (0.97)
        - Prepares sequence data for Transformer-based model
        - Uses fixed validation set from val.csv if available, otherwise supports GroupKFold
        - No external data augmentation applied
        """