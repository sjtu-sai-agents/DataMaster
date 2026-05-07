import os
import re
import string
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# --------------------------------------------------------------
# Global cache for normalization
# --------------------------------------------------------------
norm_cache = {}

# --------------------------------------------------------------
# Punctuation definition
# --------------------------------------------------------------
PUNCT_SET = set(string.punctuation)
PUNCT_SET.update(['...', '..', '....', '–', '—', '"', '"', ''', ''', '…', '``', "''", '`', "'", '"', '"', '"'])

def is_punct_token(tok):
    if tok is None:
        return True
    if not isinstance(tok, str):
        tok = str(tok)
    if not tok:
        return True
    if tok.isspace():
        return True
    if tok in PUNCT_SET:
        return True
    if all(ch in PUNCT_SET for ch in tok):
        return True
    return False

# --------------------------------------------------------------
# Context addition (skip punctuation)
# --------------------------------------------------------------
def add_prev_next_non_punct(df, token_col='before'):
    df = df.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)
    # forward pass for previous non-punct
    prev_non_punct = [None] * len(df)
    last_non_punct = None
    for i, row in df.iterrows():
        tok = row[token_col]
        if not is_punct_token(tok):
            last_non_punct = tok
        prev_non_punct[i] = last_non_punct
    # backward pass for next non-punct
    next_non_punct = [None] * len(df)
    last_non_punct = None
    for i in range(len(df)-1, -1, -1):
        tok = df.iloc[i][token_col]
        if not is_punct_token(tok):
            last_non_punct = tok
        next_non_punct[i] = last_non_punct
    df = df.copy()
    df['prev_non_punct'] = prev_non_punct
    df['next_non_punct'] = next_non_punct
    return df

# --------------------------------------------------------------
# Rule-based normalization
# --------------------------------------------------------------
DIGIT_WORDS = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
    15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
    19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty',
    50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'
}

def _num_to_words(n):
    """Convert integer n (>=0) to English words."""
    if n == 0:
        return DIGIT_WORDS[0]
    parts = []
    if n >= 1000000000:
        parts.append(_num_to_words(n // 1000000000) + " billion")
        n %= 1000000000
    if n >= 1000000:
        parts.append(_num_to_words(n // 1000000) + " million")
        n %= 1000000
    if n >= 1000:
        parts.append(_num_to_words(n // 1000) + " thousand")
        n %= 1000
    if n >= 100:
        parts.append(_num_to_words(n // 100) + " hundred")
        n %= 100
    if n > 0:
        if n < 20:
            parts.append(DIGIT_WORDS[n])
        else:
            tens = n // 10 * 10
            ones = n % 10
            if ones == 0:
                parts.append(DIGIT_WORDS[tens])
            else:
                parts.append(DIGIT_WORDS[tens] + " " + DIGIT_WORDS[ones])
    return " ".join(parts)

def _ordinal_to_words(n):
    """Convert integer n to ordinal words."""
    if n == 0:
        return "zeroth"
    card = _num_to_words(n).split()
    last = card[-1]
    if last == 'one':
        card[-1] = 'first'
    elif last == 'two':
        card[-1] = 'second'
    elif last == 'three':
        card[-1] = 'third'
    elif last == 'five':
        card[-1] = 'fifth'
    elif last == 'eight':
        card[-1] = 'eighth'
    elif last == 'nine':
        card[-1] = 'ninth'
    elif last == 'twelve':
        card[-1] = 'twelfth'
    elif last.endswith('ty'):
        card[-1] = last[:-2] + 'tieth'
    else:
        card[-1] = last + 'th'
    return ' '.join(card)

def _year_to_words(s):
    """Convert a 4-digit year string to spoken form."""
    y = int(s)
    if 1000 <= y <= 1999:
        if y % 100 == 0:
            return _num_to_words(y // 100) + " hundred"
        else:
            first = y // 100
            second = y % 100
            if second < 10:
                return _num_to_words(first) + " oh " + _num_to_words(second)
            else:
                return _num_to_words(first) + " " + _num_to_words(second)
    elif 2000 <= y <= 2099:
        if y == 2000:
            return "two thousand"
        elif y <= 2009:
            return "two thousand " + _num_to_words(y % 10)
        else:
            first = y // 100
            second = y % 100
            if second < 10:
                return _num_to_words(first) + " oh " + _num_to_words(second)
            else:
                return _num_to_words(first) + " " + _num_to_words(second)
    elif 2100 <= y <= 9999:
        if y % 100 == 0:
            return _num_to_words(y // 100) + " hundred"
        else:
            first = y // 100
            second = y % 100
            if second < 10:
                return _num_to_words(first) + " oh " + _num_to_words(second)
            else:
                return _num_to_words(first) + " " + _num_to_words(second)
    else:
        return _num_to_words(y)

def _normalize_number(num_str):
    """Convert a numeric string (with optional commas and decimal) to words."""
    num_str = num_str.replace(',', '')
    if '.' in num_str:
        int_part, frac_part = num_str.split('.')
        int_w = _num_to_words(int(int_part)) if int_part else 'zero'
        frac_w = ' '.join(DIGIT_WORDS[int(d)] for d in frac_part if d.isdigit())
        return f"{int_w} point {frac_w}" if frac_w else int_w
    else:
        return _num_to_words(int(num_str))

# Maps for abbreviations, months, days
ABBREV_TITLES_DOT = {
    'dr.': 'doctor', 'dr': 'doctor',
    'mr.': 'mister', 'mr': 'mister',
    'mrs.': 'missus', 'mrs': 'missus',
    'ms.': 'miss', 'ms': 'miss',
    'prof.': 'professor', 'prof': 'professor',
    'rev.': 'reverend', 'rev': 'reverend',
    'gov.': 'governor', 'gov': 'governor',
    'sen.': 'senator', 'sen': 'senator',
    'rep.': 'representative', 'rep': 'representative',
    'gen.': 'general', 'gen': 'general',
    'col.': 'colonel', 'col': 'colonel',
    'lt.': 'lieutenant', 'lt': 'lieutenant',
    'sgt.': 'sergeant', 'sgt': 'sergeant',
    'capt.': 'captain', 'capt': 'captain',
    'adm.': 'admiral', 'adm': 'admiral',
    'pres.': 'president', 'pres': 'president',
    'hon.': 'honorable', 'hon': 'honorable'
}
MONTHS = {
    'jan': 'january', 'jan.': 'january',
    'feb': 'february', 'feb.': 'february',
    'mar': 'march', 'mar.': 'march',
    'apr': 'april', 'apr.': 'april',
    'may': 'may', 'may.': 'may',
    'jun': 'june', 'jun.': 'june',
    'jul': 'july', 'jul.': 'july',
    'aug': 'august', 'aug.': 'august',
    'sep': 'september', 'sep.': 'september',
    'oct': 'october', 'oct.': 'october',
    'nov': 'november', 'nov.': 'november',
    'dec': 'december', 'dec.': 'december'
}
DAYS = {
    'mon': 'monday', 'mon.': 'monday',
    'tue': 'tuesday', 'tue.': 'tuesday',
    'wed': 'wednesday', 'wed.': 'wednesday',
    'thu': 'thursday', 'thu.': 'thursday',
    'fri': 'friday', 'fri.': 'friday',
    'sat': 'saturday', 'sat.': 'saturday',
    'sun': 'sunday', 'sun.': 'sunday'
}

UNIT_SINGULAR = {
    'ft': 'foot', 'in': 'inch', 'lb': 'pound', 'lbs': 'pound',
    'oz': 'ounce', 'mi': 'mile', 'km': 'kilometer', 'm': 'meter',
    'cm': 'centimeter', 'mm': 'millimeter', 'kg': 'kilogram',
    'g': 'gram', 'mg': 'milligram', 'pt': 'pint', 'qt': 'quart',
    'gal': 'gallon', 'l': 'liter', 'ml': 'milliliter',
    'tsp': 'teaspoon', 'tbsp': 'tablespoon',
    'hr': 'hour', 'h': 'hour', 'min': 'minute', 'sec': 'second', 's': 'second'
}
UNIT_PLURAL = {
    'ft': 'feet', 'in': 'inches', 'lb': 'pounds', 'lbs': 'pounds',
    'oz': 'ounces', 'mi': 'miles', 'km': 'kilometers', 'm': 'meters',
    'cm': 'centimeters', 'mm': 'millimeters', 'kg': 'kilograms',
    'g': 'grams', 'mg': 'milligrams', 'pt': 'pints', 'qt': 'quarts',
    'gal': 'gallons', 'l': 'liters', 'ml': 'milliliters',
    'tsp': 'teaspoons', 'tbsp': 'tablespoons',
    'hr': 'hours', 'h': 'hours', 'min': 'minutes', 'sec': 'seconds', 's': 'seconds'
}

FRACTION_MAP = {
    '1/2': 'one half', '1/3': 'one third', '2/3': 'two thirds',
    '1/4': 'one quarter', '3/4': 'three quarters',
    '1/5': 'one fifth', '2/5': 'two fifths', '3/5': 'three fifths', '4/5': 'four fifths',
    '1/6': 'one sixth', '5/6': 'five sixths',
    '1/8': 'one eighth', '3/8': 'three eighths', '5/8': 'five eighths', '7/8': 'seven eighths'
}

# Regex patterns
_currency_re = re.compile(r'^\$([0-9,]+(?:\.\d+)?)$')
_percent_re = re.compile(r'^([0-9,]+(?:\.\d+)?)%$')
_measure_re = re.compile(r'^([0-9,]+(?:\.\d+)?)([a-zA-Z]+)$')
_time_re = re.compile(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?$', re.I)
_fraction_re = re.compile(r'^(\d+)/(\d+)$')
_ordinal_re = re.compile(r'^(\d+)(st|nd|rd|th)$')
_number_re = re.compile(r'^[0-9,]+(?:\.\d+)?$')
_leading_decimal_re = re.compile(r'^\.(\d+)$')
_acronym_dot_re = re.compile(r'^([A-Z]\.)+[A-Z]?\.?$')
_acronym_nodot_re = re.compile(r'^[A-Z]{2,}$')
_year_re = re.compile(r'^\d{4}$')

def normalize_token(s):
    """Rule-based normalization of a single token."""
    global norm_cache
    if s in norm_cache:
        return norm_cache[s]
    original = s
    s = s.strip()
    if not s:
        norm_cache[original] = ''
        return ''
    if is_punct_token(s):
        norm_cache[original] = s
        return s
    s_lower = s.lower()
    if s_lower in ABBREV_TITLES_DOT:
        norm_cache[original] = ABBREV_TITLES_DOT[s_lower]
        return norm_cache[original]
    if s_lower in MONTHS:
        norm_cache[original] = MONTHS[s_lower]
        return norm_cache[original]
    if s_lower in DAYS:
        norm_cache[original] = DAYS[s_lower]
        return norm_cache[original]
    m = _currency_re.match(s)
    if m:
        amt = m.group(1).replace(',', '')
        if '.' in amt:
            dollars, cents = amt.split('.')
            dollars = int(dollars) if dollars else 0
            cents = int(cents.ljust(2, '0')[:2])
        else:
            dollars = int(amt)
            cents = 0
        dollar_word = _num_to_words(dollars)
        cent_word = _num_to_words(cents)
        if dollars == 0 and cents == 0:
            out = 'zero dollars'
        elif dollars == 0:
            out = f"{cent_word} cent{'s' if cents != 1 else ''}"
        elif cents == 0:
            out = f"{dollar_word} dollar{'s' if dollars != 1 else ''}"
        else:
            out = f"{dollar_word} dollar{'s' if dollars != 1 else ''} and {cent_word} cent{'s' if cents != 1 else ''}"
        norm_cache[original] = out
        return out
    m = _percent_re.match(s)
    if m:
        num = m.group(1)
        num_words = _normalize_number(num)
        out = f"{num_words} percent"
        norm_cache[original] = out
        return out
    m = _measure_re.match(s)
    if m:
        num_part, unit_part = m.groups()
        unit_lower = unit_part.lower()
        if unit_lower in UNIT_SINGULAR:
            num_words = _normalize_number(num_part)
            num_clean = num_part.replace(',', '')
            try:
                num_val = float(num_clean)
                if abs(num_val - 1.0) < 1e-6:
                    unit = UNIT_SINGULAR[unit_lower]
                else:
                    unit = UNIT_PLURAL[unit_lower]
            except:
                unit = UNIT_PLURAL[unit_lower]
            out = f"{num_words} {unit}"
            norm_cache[original] = out
            return out
    m = _time_re.match(s)
    if m:
        hour, minute, sec, ampm = m.groups()
        hour = int(hour)
        minute = int(minute)
        sec = int(sec) if sec else None
        if ampm:
            ampm = ampm.replace('.', '').lower()
            hour12 = hour % 12
            if hour12 == 0:
                hour12 = 12
            hour_word = _num_to_words(hour12)
        else:
            if hour == 0:
                hour_word = 'twelve'
            elif hour <= 12:
                hour_word = _num_to_words(hour)
            else:
                hour_word = _num_to_words(hour)
        if minute == 0 and sec is None:
            time_str = hour_word + " o'clock"
        else:
            if minute == 0:
                minute_str = ''
            elif minute < 10:
                minute_str = " oh " + _num_to_words(minute)
            else:
                minute_str = " " + _num_to_words(minute)
            time_str = hour_word + minute_str
            if sec is not None:
                time_str += " and " + _num_to_words(sec) + " second" + ('' if sec == 1 else 's')
        if ampm:
            time_str += " " + ampm[0] + " m"
        norm_cache[original] = time_str.strip()
        return norm_cache[original]
    m = _fraction_re.match(s)
    if m:
        if s in FRACTION_MAP:
            out = FRACTION_MAP[s]
            norm_cache[original] = out
            return out
        num, den = int(m.group(1)), int(m.group(2))
        num_word = _num_to_words(num)
        if den == 2:
            den_word = "half" if num == 1 else "halves"
        elif den == 3:
            den_word = "third" if num == 1 else "thirds"
        elif den == 4:
            den_word = "quarter" if num == 1 else "quarters"
        else:
            den_ord = _ordinal_to_words(den)
            if num == 1:
                den_word = den_ord
            else:
                if den_ord.endswith('th'):
                    den_word = den_ord[:-2] + 'ths'
                else:
                    den_word = den_ord + 's'
        out = f"{num_word} {den_word}"
        norm_cache[original] = out
        return out
    m = _ordinal_re.match(s)
    if m:
        num = int(m.group(1))
        out = _ordinal_to_words(num)
        norm_cache[original] = out
        return out
    m = _leading_decimal_re.match(s)
    if m:
        frac = m.group(1)
        frac_words = ' '.join(DIGIT_WORDS[int(d)] for d in frac)
        out = "point " + frac_words
        norm_cache[original] = out
        return out
    m = _year_re.match(s)
    if m and 1000 <= int(s) <= 9999:
        out = _year_to_words(s)
        norm_cache[original] = out
        return out
    if _number_re.match(s):
        out = _normalize_number(s)
        norm_cache[original] = out
        return out
    if _acronym_dot_re.match(s):
        letters = [ch for ch in s if ch.isalpha()]
        out = ' '.join(letters).lower()
        norm_cache[original] = out
        return out
    if _acronym_nodot_re.match(s):
        out = ' '.join(s).lower()
        norm_cache[original] = out
        return out
    norm_cache[original] = s
    return s


class MyDataLoader(BaseDataLoader):
    """
    Data loader for text normalization task.
    Loads training and test data, computes rule-based normalization statistics,
    and builds context-based lookup tables for tokens where rules fail.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dir = kwargs.get('input_dir', './input')
        
    def setup(self):
        """
        Load data, compute statistics, and build lookup tables.
        Sets self.train_data and self.test_data.
        """
        global norm_cache
        
        # Load test data
        test_df = pd.read_csv(
            os.path.join(self.input_dir, "en_test.csv"),
            usecols=["sentence_id", "token_id", "before"],
            dtype={"sentence_id": int, "token_id": int, "before": str}
        )
        test_df["before"] = test_df["before"].astype(str)
        
        # Initialize counters
        total_counts = {}
        rb_correct_counts = {}
        bad_befores = set()
        val_rows = []
        
        # Check if val.csv exists
        val_csv_path = os.path.join(self.input_dir, 'val.csv')
        if os.path.exists(val_csv_path):
            val_df_pre = pd.read_csv(val_csv_path)
            val_df_pre["before"] = val_df_pre["before"].astype(str)
            val_df_pre["after"] = val_df_pre["after"].astype(str)
            val_sentences = set(val_df_pre['sentence_id'].unique())
        else:
            val_sentences = None
        
        # First pass: read training, split validation, compute rule-based correctness
        train_chunks = pd.read_csv(
            os.path.join(self.input_dir, "en_train.csv"),
            chunksize=500000,
            usecols=["sentence_id", "token_id", "before", "after"],
            dtype={"sentence_id": int, "token_id": int, "before": str, "after": str}
        )
        for chunk_idx, chunk in enumerate(train_chunks):
            chunk["before"] = chunk["before"].astype(str)
            chunk["after"] = chunk["after"].astype(str)
            
            # Split validation
            if val_sentences is not None:
                mask_val = chunk["sentence_id"].isin(val_sentences)
            else:
                mask_val = (chunk["sentence_id"] % 50 == 0)
            
            val_chunk = chunk[mask_val]
            train_chunk = chunk[~mask_val]
            
            # Collect validation rows
            for _, row in val_chunk.iterrows():
                val_rows.append({
                    "sentence_id": row.sentence_id,
                    "token_id": row.token_id,
                    "before": row.before,
                    "after": row.after
                })
            
            # Process training fold
            uniq_befores = set(train_chunk["before"].unique())
            for b in uniq_befores:
                if b not in norm_cache:
                    norm_cache[b] = normalize_token(b)
            
            # Update counts
            for b, a in zip(train_chunk["before"], train_chunk["after"]):
                total_counts[b] = total_counts.get(b, 0) + 1
                if norm_cache[b] == a:
                    rb_correct_counts[b] = rb_correct_counts.get(b, 0) + 1
                else:
                    bad_befores.add(b)
        
        # Build memorization counters
        mem_counts = defaultdict(Counter)
        left_counts = defaultdict(Counter)
        right_counts = defaultdict(Counter)
        tri_counts = defaultdict(Counter)
        
        # Second pass: read training again, add context, count for bad_befores
        train_chunks2 = pd.read_csv(
            os.path.join(self.input_dir, "en_train.csv"),
            chunksize=500000,
            usecols=["sentence_id", "token_id", "before", "after"],
            dtype={"sentence_id": int, "token_id": int, "before": str, "after": str}
        )
        for chunk_idx, chunk in enumerate(train_chunks2):
            chunk["before"] = chunk["before"].astype(str)
            chunk["after"] = chunk["after"].astype(str)
            
            # Filter to training fold (exclude validation)
            if val_sentences is not None:
                chunk = chunk[~chunk["sentence_id"].isin(val_sentences)]
            else:
                chunk = chunk[chunk["sentence_id"] % 50 != 0]
            
            if len(chunk) == 0:
                continue
            
            # Add context
            chunk = add_prev_next_non_punct(chunk)
            
            # Filter rows with before in bad_befores
            chunk_bad = chunk[chunk["before"].isin(bad_befores)]
            
            # Update counters
            for _, row in chunk_bad.iterrows():
                b = row["before"]
                a = row["after"]
                prev = row["prev_non_punct"]
                next_ = row["next_non_punct"]
                mem_counts[b][a] += 1
                if prev is not None:
                    left_counts[(prev, b)][a] += 1
                if next_ is not None:
                    right_counts[(b, next_)][a] += 1
                if prev is not None and next_ is not None:
                    tri_counts[(prev, b, next_)][a] += 1
        
        # Build validation dataframe
        val_df = pd.DataFrame(val_rows)
        if len(val_df) == 0:
            val_df = pd.read_csv(
                os.path.join(self.input_dir, "en_train.csv"),
                nrows=1000,
                usecols=["sentence_id", "token_id", "before", "after"]
            )
            val_df["before"] = val_df["before"].astype(str)
            val_df["after"] = val_df["after"].astype(str)
        
        # Set train_data and test_data
        self.train_data = {
            'total_counts': total_counts,
            'rb_correct_counts': rb_correct_counts,
            'bad_befores': bad_befores,
            'mem_counts': dict(mem_counts),
            'left_counts': dict(left_counts),
            'right_counts': dict(right_counts),
            'tri_counts': dict(tri_counts),
            'val_df': val_df,
            'norm_cache': dict(norm_cache)
        }
        self.test_data = test_df
        
    def describe(self) -> str:
        """
        Return a description of the data processing approach.
        """
        return ("Text normalization data loader with rule-based normalization and context-based features. "
                "Uses memorization and trigram context for tokens where rule-based approach fails. "
                "Supports fixed validation set from input/val.csv or deterministic split (sentence_id % 50 == 0).")