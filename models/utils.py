import nltk
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("omw-1.4")

import os
import regex as re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

max_sentence_length = 1551

def remove_stopwords(text:str):
    '''
    This function removes english stopwords from input text.
    
    Parameters
    -----------
        text : (str) Input text.

    Returns
    -------
    (str): Filtered text without stopwords.
    '''
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]
    words = text.split()
    filtered_text = [word for word in words if (word not in stopwords_list or word in whitelist)]
    return ' '.join(filtered_text)

def remove_breaklines(text:str):
    '''
    This function removes 'br' from input text.
    
    Parameters
    -----------
        text : (str) Input text.

    Returns
    -------
    (str): Filtered text without 'br'.
    '''
    words = text.split()
    filtered_text = [word.replace('br', '') for word in words]
    return ' '.join(filtered_text)

def clean_text(text: str):
    '''
    Cleans the input text by removing all punctuation, numbers, and special symbols,
    leaving only alphabetic characters and whitespace.

    This function uses a regular expression to replace all characters that are not
    letters (a-z, A-Z) or whitespace.

    Parameters
    ----------
    text (str): The input string to be cleaned.

    Returns
    -------
    (str): The cleaned string with only alphabetic characters and whitespace.
    '''
    pattern = r'[^a-zA-Z\s]'
    return re.sub(pattern, ' ', text)

def lemmatize(text: str):
    '''
    This functions lemmatizes the input text by reducing each word to its base form (lemma).

    Parameters
    ----------
    text: (str) The input text to be lemmatized.

    Returns
    -------
    (str): The lemmatized text where each word is reduced to its base form.
    '''
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]
    return ' '.join(lemmatized_words)

def load_data(directory, texts, labels):
    '''
    This function reads all text files in the given directory, appending the
    content of each file to the `texts` list and the corresponding rating 
    (extracted from the file name) to the `labels` list.

    Parameters
    ----------
    directory (str): The path to the directory containing the text files.
    labels (list): A list to which the ratings will be appended.
    texts (list): A list to which the contents of the text files will be appended.

    Each file in the directory should have a name in the format 'name_rating.txt'.
    
    There are two possible ratings:
    - Negative (0): For ratings with a value of 4 or less.
    - Positive (1): For ratings with a value of 5 or more.
    '''
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as file:
            texts.append(file.read())
        rating = file_name.split('_')[1].strip('.txt')
        labels.append(0 if int(rating) <= 4 else 1)