import numpy as np
import pandas as pd
import re

training_data =pd.read_csv('ted_main.csv')


def get_words_title(message):
    new_message = []
    if "?" in message:
        new_message.append("?")
    if "!" in message:
        new_message.append("!")
    message = re.sub("[^\w\s]", "", message)
    new_message += [word.lower() for word in message.split(' ')]
    return new_message


def get_words_tags(message):
    message = message[2:-2]
    return [word.lower() for word in message.split("', '")]

def create_dictionary(messages,data_type):
    if data_type == 'title':
        get_words = get_words_title
    if data_type == 'tags':
        get_words = get_words_tags
    all_words = {}
    final_words = {}
    for message in messages:
        word_list = set(get_words(message))
        for word in word_list:
            if word in all_words.keys():
                all_words[word] += 1
            else:
                all_words[word] = 1
    for word in all_words.keys():
        if all_words[word] >= 5:
            final_words[word]=len(final_words)
    return final_words

def transform_text(messages, word_dictionary):
    n = len(word_dictionary)
    m = len(messages)
    array_words = np.zeros((m, n))

    for j in range(m):
        message = messages[j]
        list_words = get_words(message)
        for word in list_words:
            if word in word_dictionary.keys():
                i = word_dictionary[word]
                array_words[j, i] += 1

    return array_words

train_titles = list(training_data['title'])
train_tags = list(training_data['tags'])
train_description = list(training_data['description'])

dictionary_titles = create_dictionary(train_titles,'title')
dictionary_tags = create_dictionary(train_tags,'tags')
dictionary_description = create_dictionary(train_description,'title')