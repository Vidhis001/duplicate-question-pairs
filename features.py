import re
import nltk
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz

# A helper for safe division to avoid ZeroDivisionError
SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words('english')

def common_words(q1, q2):
    """Calculates the number of common words between two questions."""
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def total_words(q1, q2):
    """Calculates the total number of unique words between two questions."""
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1) + len(w2)

def fetch_token_features(q1, q2):
    """
    Generates a list of 8 advanced token-based features.
    """
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

def fetch_length_features(q1, q2):
    """Generates a list of 3 length-based features."""
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    if len(strs) > 0:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0.0

    return length_features

def fetch_fuzzy_features(q1, q2):
    """Generates a list of 4 fuzzy-based features."""
    fuzzy_features = [0.0] * 4
    
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    
    return fuzzy_features

def generate_all_features(q1, q2):
    """
    Combines all feature engineering functions to create a single list of features.
    """
    features = []
    
    # Simple Features
    features.append(len(q1))
    features.append(len(q2))
    features.append(len(q1.split(" ")))
    features.append(len(q2.split(" ")))
    
    common = common_words(q1, q2)
    total = total_words(q1, q2)
    features.append(common)
    features.append(total)
    features.append(round(common / (total + SAFE_DIV), 2))

    # Advanced Token Features
    features.extend(fetch_token_features(q1, q2))

    # Length Features
    features.extend(fetch_length_features(q1, q2))

    # Fuzzy Features
    features.extend(fetch_fuzzy_features(q1, q2))
    
    return features
