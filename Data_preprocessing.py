import re
import numpy as np

from transformers import BertTokenizer
## Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_words(message,type_data, tokenizing = False):
    """
    This function gets words / tokens of the titles.
    If tokenizing == True, it splits titles into token.
    Otherwise, it splits a title into words, normalize them, and return
    the resulting list. We are splitting on spaces. And we are keeping
    ! and ?.
    """
    if tokenizing == True:
        new_message = tokenizer.tokenize(message)
        return new_message
    else:
        if type_data in ['title', 'description']:
            new_message = []
            if "?" in message:
                new_message.append("?")
            if "!" in message:
                new_message.append("!")
            message = re.sub("[^\w\s]", "", message)
            new_message += [word.lower() for word in message.split(' ')]
            return new_message

        if type_data == 'tags':
            message = message[2:-2]
            return [word.lower() for word in message.split("', '")]

def create_dictionary(messages, type_data, tokenizing=False):
    """
    This function creates a dictionary of word to indices using the provided
    training messages.
    We are only adding words if they occur in at least 3 messages.
    """

    CompleteWordList = []
    WordCounter = []

    for message in messages:
        word_list = set(get_words(message, type_data, tokenizing))  # get rid of duplicates
        for word in word_list:
            if word in CompleteWordList:
                incrementationIndex = CompleteWordList.index(word)
                WordCounter[incrementationIndex] += 1
            else:
                CompleteWordList.append(word)
                WordCounter.append(1)

    dictionary = dict()

    counter = 0
    for wordnumber in range(len(CompleteWordList)):
        if WordCounter[wordnumber] >= 3:
            dictionary[CompleteWordList[wordnumber]] = counter
            counter += 1
    return dictionary


def transform_text(messages, word_dictionary, type_data, tokenizing=False):
    """
    This function creates a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array corresponds to each message
    and each column corresponds to a word of the vocabulary.
    """

    numberMessages = len(messages)
    numberWords = len(word_dictionary)
    result = np.zeros((numberMessages, numberWords))
    for i in range(numberMessages):
        message = messages.values[i]
        wordList = get_words(message, type_data, tokenizing)
        for word in wordList:
            if word in word_dictionary:
                word_index = word_dictionary[word]
                result[i, word_index] += 1
    return result


