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