
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')


ps = PorterStemmer()

def preprocess(q):
    q = str(q).lower().strip()
    
    # Replace special symbols
    q = q.replace('%', ' percent').replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')

    # Replace numbers
    q = q.replace(',000,000,000 ', 'b ').replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontract words
    contractions = {
        "can't": "can not", "won't": "will not", "i'm": "i am",
        "it's": "it is", "don't": "do not", "i've": "i have",
        "you're": "you are", "i'll": "i will", "isn't": "is not"
    }
    q = ' '.join([contractions.get(word, word) for word in q.split()])

    # Remove HTML
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove punctuation
    q = re.sub(r'\W', ' ', q).strip()

    # Apply stemming
    stemmed = [ps.stem(word) for word in q.split()]
    q = " ".join(stemmed)

    return q
