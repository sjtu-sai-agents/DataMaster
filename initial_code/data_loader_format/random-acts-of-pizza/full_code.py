import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def extract_text_features(text):
    """Extract 9 advanced text features from a string."""
    if not text or pd.isna(text):
        return [0.0] * 9
    try:
        sent = analyzer.polarity_scores(text)
    except:
        sent = {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    try:
        flesch = textstat.flesch_reading_ease(text)
        kincaid = textstat.flesch_kincaid_grade(text)
    except:
        flesch = 0.0
        kincaid = 0.0
    excl = text.count('!')
    quest = text.count('?')
    if len(text) > 0:
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    else:
        upper_ratio = 0.0
    return [sent['compound'], sent['pos'], sent['neu'], sent['neg'],
            flesch, kincaid, excl, quest, upper_ratio]


class MyDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = None
        
    def setup(self):
        """
        Load data, perform feature engineering, and prepare train/test sets.
        Uses fixed validation set from input/val.csv if available.
        """
        print("Loading data...")
        with open('./input/train/train.json', 'r') as f:
            train = pd.DataFrame(json.load(f))
        with open('./input/test/test.json', 'r') as f:
            test = pd.DataFrame(json.load(f))
        
        # Check for fixed validation set
        val_df = None
        if os.path.exists('input/val.csv'):
            val_df = pd.read_csv('input/val.csv')
            print(f"Found validation set with {len(val_df)} samples")
        
        # Target
        y = train['requester_received_pizza'].astype(int)
        
        # Store request_ids for validation split
        train_request_ids = train['request_id'].values
        
        # ---------- Combine text ----------
        train['combined_text'] = train['request_title'].fillna('') + ' ' + train['request_text_edit_aware'].fillna('')
        test['combined_text'] = test['request_title'].fillna('') + ' ' + test['request_text_edit_aware'].fillna('')
        
        # ---------- Advanced text features ----------
        adv_text_cols = ['sent_compound', 'sent_pos', 'sent_neu', 'sent_neg',
                         'flesch', 'kincaid', 'excl', 'quest', 'upper_ratio']
        
        print("Extracting advanced text features for train...")
        train_adv = train['combined_text'].apply(extract_text_features).apply(pd.Series)
        train_adv.columns = adv_text_cols
        train = pd.concat([train, train_adv], axis=1)
        
        print("Extracting advanced text features for test...")
        test_adv = test['combined_text'].apply(extract_text_features).apply(pd.Series)
        test_adv.columns = adv_text_cols
        test = pd.concat([test, test_adv], axis=1)
        
        # ---------- Basic meta features ----------
        train['giver_known'] = train['giver_username_if_known'].apply(lambda x: 0 if x == "N/A" else 1)
        test['giver_known'] = test['giver_username_if_known'].apply(lambda x: 0 if x == "N/A" else 1)
        
        # Time features from UTC timestamp
        train['req_time'] = pd.to_datetime(train['unix_timestamp_of_request_utc'], unit='s', errors='coerce')
        test['req_time'] = pd.to_datetime(test['unix_timestamp_of_request_utc'], unit='s', errors='coerce')
        train['request_hour'] = train['req_time'].dt.hour.fillna(0).astype(int)
        train['request_dayofweek'] = train['req_time'].dt.dayofweek.fillna(0).astype(int)
        test['request_hour'] = test['req_time'].dt.hour.fillna(0).astype(int)
        test['request_dayofweek'] = test['req_time'].dt.dayofweek.fillna(0).astype(int)
        train.drop('req_time', axis=1, inplace=True)
        test.drop('req_time', axis=1, inplace=True)
        
        # ---------- Length features ----------
        train['text_length'] = train['request_text_edit_aware'].fillna('').apply(len)
        test['text_length'] = test['request_text_edit_aware'].fillna('').apply(len)
        train['title_length'] = train['request_title'].fillna('').apply(len)
        test['title_length'] = test['request_title'].fillna('').apply(len)
        
        # ---------- Behavioral ratio features ----------
        eps = 1e-6
        train['comments_per_day'] = train['requester_number_of_comments_at_request'] / (train['requester_account_age_in_days_at_request'] + eps)
        test['comments_per_day'] = test['requester_number_of_comments_at_request'] / (test['requester_account_age_in_days_at_request'] + eps)
        
        train['posts_per_day'] = train['requester_number_of_posts_at_request'] / (train['requester_account_age_in_days_at_request'] + eps)
        test['posts_per_day'] = test['requester_number_of_posts_at_request'] / (test['requester_account_age_in_days_at_request'] + eps)
        
        train['raop_comments_ratio'] = train['requester_number_of_comments_in_raop_at_request'] / (train['requester_number_of_comments_at_request'] + eps)
        test['raop_comments_ratio'] = test['requester_number_of_comments_in_raop_at_request'] / (test['requester_number_of_comments_at_request'] + eps)
        
        train['raop_posts_ratio'] = train['requester_number_of_posts_on_raop_at_request'] / (train['requester_number_of_posts_at_request'] + eps)
        test['raop_posts_ratio'] = test['requester_number_of_posts_on_raop_at_request'] / (test['requester_number_of_posts_at_request'] + eps)
        
        train['upvotes_per_comment'] = train['requester_upvotes_plus_downvotes_at_request'] / (train['requester_number_of_comments_at_request'] + eps)
        test['upvotes_per_comment'] = test['requester_upvotes_plus_downvotes_at_request'] / (test['requester_number_of_comments_at_request'] + eps)
        
        train['upvotes_per_post'] = train['requester_upvotes_plus_downvotes_at_request'] / (train['requester_number_of_posts_at_request'] + eps)
        test['upvotes_per_post'] = test['requester_upvotes_plus_downvotes_at_request'] / (test['requester_number_of_posts_at_request'] + eps)
        
        train['subreddits_per_day'] = train['requester_number_of_subreddits_at_request'] / (train['requester_account_age_in_days_at_request'] + eps)
        test['subreddits_per_day'] = test['requester_number_of_subreddits_at_request'] / (test['requester_account_age_in_days_at_request'] + eps)
        
        train['text_to_title_ratio'] = train['text_length'] / (train['title_length'] + eps)
        test['text_to_title_ratio'] = test['text_length'] / (test['title_length'] + eps)
        
        # ---------- Define feature sets ----------
        base_meta_cols = [
            'requester_account_age_in_days_at_request',
            'requester_days_since_first_post_on_raop_at_request',
            'requester_number_of_comments_at_request',
            'requester_number_of_comments_in_raop_at_request',
            'requester_number_of_posts_at_request',
            'requester_number_of_posts_on_raop_at_request',
            'requester_number_of_subreddits_at_request',
            'requester_upvotes_minus_downvotes_at_request',
            'requester_upvotes_plus_downvotes_at_request',
            'unix_timestamp_of_request',
            'unix_timestamp_of_request_utc',
            'giver_known',
            'request_hour',
            'request_dayofweek'
        ]
        
        ratio_cols = [
            'comments_per_day',
            'posts_per_day',
            'raop_comments_ratio',
            'raop_posts_ratio',
            'upvotes_per_comment',
            'upvotes_per_post',
            'text_length',
            'title_length',
            'text_to_title_ratio',
            'subreddits_per_day'
        ]
        
        meta_cols = base_meta_cols + adv_text_cols + ratio_cols
        
        # Fill missing values
        for col in meta_cols:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)
        
        # ---------- TF-IDF on combined text ----------
        print("Fitting TF-IDF...")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                          ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_train_tfidf = self.vectorizer.fit_transform(train['combined_text'].fillna(''))
        X_test_tfidf = self.vectorizer.transform(test['combined_text'].fillna(''))
        
        # ---------- Combine all features ----------
        X_train_meta = train[meta_cols].values
        X_test_meta = test[meta_cols].values
        
        X_train = hstack([X_train_tfidf, X_train_meta]).tocsr()
        X_test = hstack([X_test_tfidf, X_test_meta]).tocsr()
        
        # ---------- Handle validation set ----------
        if val_df is not None:
            # Get validation request IDs
            if 'request_id' in val_df.columns:
                val_ids = set(val_df['request_id'].values)
            else:
                # Fallback: assume first column contains IDs
                val_ids = set(val_df.iloc[:, 0].values)
            
            # Create masks for splitting
            val_mask = np.array([rid in val_ids for rid in train_request_ids])
            train_mask = ~val_mask
            
            # Split data
            X_val = X_train[val_mask]
            y_val = y[val_mask].values
            X_train_final = X_train[train_mask]
            y_train_final = y[train_mask].values
            
            print(f"Train samples: {X_train_final.shape[0]}, Validation samples: {X_val.shape[0]}")
            
            self.train_data = {
                'X': X_train_final,
                'y': y_train_final,
                'X_val': X_val,
                'y_val': y_val,
                'has_val': True
            }
        else:
            self.train_data = {
                'X': X_train,
                'y': y.values,
                'has_val': False
            }
        
        self.test_data = {
            'X': X_test,
            'ids': test['request_id'].values
        }
        
    def describe(self) -> str:
        """
        Return a description of the data processing approach.
        """
        return """
        Data processing approach for RAOP (Random Acts of Pizza) prediction:
        
        1. Data Loading:
           - Load train.json and test.json from input directory
           - Use fixed validation set from input/val.csv if available
        
        2. Text Feature Engineering:
           - Combine request_title and request_text_edit_aware into combined_text
           - Extract 9 advanced text features using VADER sentiment analysis and textstat:
             * Sentiment scores: compound, positive, neutral, negative
             * Readability: Flesch reading ease, Flesch-Kincaid grade
             * Punctuation: exclamation marks, question marks
             * Uppercase ratio
        
        3. Meta Features:
           - Giver known indicator (binary)
           - Time features: request hour, day of week
           - Length features: text length, title length
           - Behavioral ratio features:
             * comments_per_day, posts_per_day
             * raop_comments_ratio, raop_posts_ratio
             * upvotes_per_comment, upvotes_per_post
             * subreddits_per_day, text_to_title_ratio
        
        4. Text Vectorization:
           - TF-IDF on combined_text with max_features=5000
           - ngram_range=(1,2), stop_words='english'
           - min_df=2, max_df=0.95
        
        5. Feature Combination:
           - Concatenate TF-IDF sparse matrix with dense meta features
           - Handle missing values by filling with 0
        """

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LightGBM training for RAOP prediction')
    
    # Model hyperparameters
    parser.add_argument('--num_leaves', type=int, default=31,
                        help='Number of leaves in LightGBM (default: 31)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate for boosting (default: 0.05)')
    parser.add_argument('--feature_fraction', type=float, default=0.9,
                        help='Feature fraction for each tree (default: 0.9)')
    parser.add_argument('--bagging_fraction', type=float, default=0.8,
                        help='Bagging fraction for each tree (default: 0.8)')
    parser.add_argument('--bagging_freq', type=int, default=5,
                        help='Bagging frequency (default: 5)')
    parser.add_argument('--num_boost_round', type=int, default=1000,
                        help='Number of boosting rounds (default: 1000)')
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                        help='Early stopping rounds (default: 50)')
    
    # Training parameters
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV splits when no fixed validation set (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='Device to use for training (default: gpu)')
    
    # Paths
    parser.add_argument('--output_path', type=str, default='./submission/submission.csv',
                        help='Output path for submission file (default: ./submission/submission.csv)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Ensure submission directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load data using MyDataLoader
    print("Initializing data loader...")
    data_loader = MyDataLoader()
    train_data, test_data = data_loader.get_data()
    
    # Extract data
    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    test_ids = test_data['ids']
    has_val = train_data.get('has_val', False)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': args.bagging_freq,
        'verbose': -1,
        'num_threads': -1,
        'seed': args.seed,
        'device': args.device,
    }
    
    if has_val:
        # Use fixed validation set
        X_val = train_data['X_val']
        y_val = train_data['y_val']
        
        print("\nTraining with fixed validation set...")
        print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        # Calculate scale_pos_weight for imbalanced data
        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        params_fold = params.copy()
        params_fold['scale_pos_weight'] = scale_pos_weight
        
        model = lgb.train(
            params_fold, 
            lgb_train, 
            num_boost_round=args.num_boost_round,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)]
        )
        
        val_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, val_pred)
        print(f"Validation AUC: {auc:.5f}")
        
        test_preds = model.predict(X_test)
        
    else:
        # Use cross-validation
        print("\nStarting cross-validation...")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        
        fold_aucs = []
        test_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\nFold {fold+1}/{args.n_splits}")
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            lgb_train = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            
            # Calculate scale_pos_weight for imbalanced data
            pos = y_tr.sum()
            neg = len(y_tr) - pos
            scale_pos_weight = neg / pos if pos > 0 else 1.0
            params_fold = params.copy()
            params_fold['scale_pos_weight'] = scale_pos_weight
            
            model = lgb.train(
                params_fold, 
                lgb_train, 
                num_boost_round=args.num_boost_round,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)]
            )
            
            val_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, val_pred)
            fold_aucs.append(auc)
            print(f"Fold {fold+1} AUC: {auc:.5f}")
            
            test_preds += model.predict(X_test)
        
        test_preds /= args.n_splits
        
        cv_mean = np.mean(fold_aucs)
        cv_std = np.std(fold_aucs)
        print(f"\nCV AUC: {cv_mean:.5f} ± {cv_std:.5f}")
    
    # Save submission
    submission = pd.DataFrame({
        'request_id': test_ids,
        'requester_received_pizza': test_preds
    })
    submission.to_csv(args.output_path, index=False)
    print(f"\nSubmission saved to {args.output_path}")


if __name__ == "__main__":
    main()