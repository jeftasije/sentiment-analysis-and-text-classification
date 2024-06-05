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


def preprocess_text(text: str):
    '''
    This function preprocesses the input text by performing the following steps:
    1. Removes stopwords except for "n't", "not", and "no".
    2. Removes 'br' tags from the text.
    3. Cleans the text by removing all punctuation, numbers, and special symbols.
    4. Lemmatizes the text by reducing each word to its base form.

    Parameters
    ----------
    text : (str)
        The input text to be preprocessed.

    Returns
    -------
    (str): The preprocessed text.
    '''
    
    def remove_stopwords(text: str):
        '''
        This function removes English stopwords from the input text.
        
        Parameters
        -----------
        text : (str) 
            Input text.
        
        Returns
        -------
        (str): Filtered text without stopwords.
        '''
        stopwords_list = stopwords.words('english')
        whitelist = ["n't", "not", "no"]
        words = text.split()
        filtered_text = [word for word in words if (word not in stopwords_list or word in whitelist)]
        return ' '.join(filtered_text)

    def remove_breaklines(text: str):
        '''
        This function removes 'br' from the input text.
        
        Parameters
        -----------
        text : (str) 
            Input text.
        
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
        
        Parameters
        ----------
        text : (str) 
            The input string to be cleaned.
        
        Returns
        -------
        (str): The cleaned string with only alphabetic characters and whitespace.
        '''
        pattern = r'[^a-zA-Z\s]'
        return re.sub(pattern, ' ', text)

    def lemmatize(text: str):
        '''
        This function lemmatizes the input text by reducing each word to its base form (lemma).
        
        Parameters
        ----------
        text : (str) 
            The input text to be lemmatized.
        
        Returns
        -------
        (str): The lemmatized text where each word is reduced to its base form.
        '''
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]
        return ' '.join(lemmatized_words)


    text = remove_stopwords(text)
    text = remove_breaklines(text)
    text = clean_text(text)
    text = lemmatize(text)
    
    return text

def load_data_from_folder(folder_path):
    '''
    This function loads text data from a folder where each subfolder represents a category.
    Each subfolder contains text files corresponding to that category.

    Parameters
    ----------
    folder_path : (str)
        The path to the folder containing the dataset.

    Returns
    -------
    data : (list)
        A list of texts.
    target : (list)
        A list of labels corresponding to each text.
    target_names : (list)
        A list of category names.
    '''
    data = []
    target = []
    target_names = os.listdir(folder_path)
    target_names.sort()  
    label_map = {name: idx for idx, name in enumerate(target_names)}

    for category in target_names:
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='latin1') as file:
                        text = file.read()
                        text = preprocess_text(text)  
                        data.append(text)
                        target.append(label_map[category])

    return data, target, target_names

def RNN_preprocess_text(text: str):
    '''
    This function preprocesses the input text by performing the following steps:
    1. Cleans the text by removing all punctuation, numbers, and special symbols.
    2. Converts text to lowercase.

    Parameters
    ----------
    text : (str)
        The input text to be preprocessed.

    Returns
    -------
    (str): The preprocessed text.
    '''

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()
    return text

def RNN_load_data_from_folder(folder_path):
    '''
    This function loads text data from a folder where each subfolder represents a category.
    Each subfolder contains text files corresponding to that category.

    Parameters
    ----------
    folder_path : (str)
        The path to the folder containing the dataset.

    Returns
    -------
    data : (list)
        A list of texts.
    target : (list)
        A list of labels corresponding to each text.
    target_names : (list)
        A list of category names.
    '''
    data = []
    target = []
    target_names = os.listdir(folder_path)
    target_names.sort() 
    label_map = {name: idx for idx, name in enumerate(target_names)}

    for category in target_names:
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='latin1') as file:
                        text = file.read()
                        text = RNN_preprocess_text(text)  
                        data.append(text)
                        target.append(label_map[category])

    return data, target, target_names