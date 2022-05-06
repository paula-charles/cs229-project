import collections

import numpy as np
import csv
import util
import pandas as pd
import matplotlib.pyplot as plt
import re


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, step_size=1e-3, max_iter=1000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient descent to maximize likelihood 

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples,dim = np.shape(x)
        reg=1
        self.theta = np.zeros(dim)
        norm_change = self.eps+1
        iteration_number=0
        while norm_change > self.eps and iteration_number<self.max_iter:
            iteration_number+=1
            hypothesis = self.theta.dot(x.T)
            
            gradient = 1/n_examples*(y-hypothesis).dot(x)-reg*self.theta
            #print(gradient)
            #gradient = np.array(sum(x[i]*(y[i]-hypothesis[i]) for i in range (n_examples)))-reg*self.theta
            self.theta += self.step_size*gradient
            norm_change = np.linalg.norm(gradient,1)
        return None
    
    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n_examples = len(X)
        augmented_X = np.zeros(shape=(n_examples,k+1))
        for i in range(len(X)):
            for j in range(k+1):
                augmented_X[i,j] = X[i,1]**j
        return augmented_X    
        # *** END CODE HERE ***
    
    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.dot(self.theta,X.T)
        # *** END CODE HERE ***

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    #matrix[0] is the word counts for the first sentence
    avg_perf = np.average(labels)
    
    n_examples,n_words= np.shape(matrix)
    
    sum_words=np.zeros(n_words)
    word_appearance = np.zeros(n_words)
    for k in range(n_examples):
        for i in range(n_words):
            sum_words[i] += matrix[k, i]*labels[k]
            word_appearance[i] += matrix[k, i]
    
    avg_perf_word = sum_words/word_appearance #I may need to do some laplace smoothing
    return (avg_perf, avg_perf_word)
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi=model[0]
    phiX_knowingY = model[1]
    
    nExamples = len(matrix)
    prediction = np.zeros(nExamples)
    for exampleIndex in range(nExamples):
        sum0 = 0
        sum1 = 0
        for wordIndex in range(len(matrix[0])):
            if matrix[exampleIndex,wordIndex] == 0:
                sum0 += np.log(1-phiX_knowingY[0][wordIndex])
                sum1 += np.log(1-phiX_knowingY[1][wordIndex])
            else:
                sum0 += matrix[exampleIndex,wordIndex]*np.log(phiX_knowingY[0][wordIndex])
                sum1 += matrix[exampleIndex,wordIndex]*np.log(phiX_knowingY[1][wordIndex])
        prob_of_1 = np.log(phi) + sum1
        prob_of_0 = np.log(1-phi) + sum0
        if prob_of_1 >= prob_of_0:
            prediction[exampleIndex] = 1
    return prediction
    
    # *** END CODE HERE ***
    


def get_words(message,type_data):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
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
    # *** END CODE HERE ***


def create_dictionary(messages,type_data):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    CompleteWordList = []
    WordCounter = []

    for message in messages:
        word_list= set(get_words(message,type_data)) #get rid of duplicates
        for word in word_list:
            if word in CompleteWordList:
                incrementationIndex = CompleteWordList.index(word)
                WordCounter[incrementationIndex]+=1
            else:
                CompleteWordList.append(word)
                WordCounter.append(1)
                
    dictionary = dict()

    counter = 0
    for wordnumber in range(len(CompleteWordList)):
        if WordCounter[wordnumber] >= 3:
            dictionary[CompleteWordList[wordnumber]]=counter
            counter+=1
    return dictionary
        
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary,type_data):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    numberMessages = len(messages)
    numberWords = len(word_dictionary)
    result = np.zeros((numberMessages,numberWords))
    for i in range(numberMessages):
        message = messages[i]
        wordList = get_words(message,type_data)
        for word in wordList:
            if word in word_dictionary:
                word_index = word_dictionary[word]
                result[i,word_index]+=1
    return result
    # *** END CODE HERE ***



def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    avg_perf_word = model[1]
    indices = []
    for k in range(5):
        ind = np.argmax(avg_perf_word)
        indices.append(ind)
        avg_perf_word[ind] = 0
    
    return [list(dictionary.keys())[indices[k]] for k in range(5)]
    
    # *** END CODE HERE ***


def main():
  
    df = pd.read_csv('ted_main.csv', delimiter=',')
    tuples = [list(x) for x in df.values]
    
    n_examples_total = len(tuples)
    n_examples = int(n_examples_total*0.7)
    n_val = int(n_examples_total*0.9)
    
    training_data = tuples[:n_examples]
    validation_data = tuples[n_examples:n_val]
    test_data = tuples[n_val:]
    
    names = [list(datapoint[7].rpartition(": "))[-1] for datapoint in training_data]
    number_comments = [datapoint[0] for datapoint in training_data]
    number_views = [datapoint[-1] for datapoint in training_data]
    
    dictionary = create_dictionary(names, 'title')
    #print(dictionary)
    print('Size of dictionary: ', len(dictionary))
    
    train_matrix = transform_text(names, dictionary,'title')
    
    linear = LinearModel()
    linear.fit(train_matrix,np.array(number_views))
    
    val_names = [list(datapoint[7].rpartition(": "))[-1] for datapoint in validation_data]
    val_number_comments = [datapoint[0] for datapoint in validation_data]
    #val_number_views = [datapoint[-1] for datapoint in validation_data]
    val_number_views = [datapoint[-1] for datapoint in training_data]

    val_matrix = transform_text(val_names, dictionary,'title')
    
    prediction = linear.predict(val_matrix)
    
    #plt.xlim(0,5*10**6)
    #plt.ylim(0,1*10**6)
    def plot_loghist(x, bins):
        hist, bins = np.histogram(x, bins=bins)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(x, bins=logbins)
        plt.xscale('log')

    plot_loghist(val_number_views, 20)
    plt.title('Number of views of the titles in the training data set\n'
              '(log-scale)')
    plt.show()
    
    
    naive_bayes_model_views = fit_naive_bayes_model(train_matrix,number_views)
    naive_bayes_model_comments = fit_naive_bayes_model(train_matrix,number_comments)
    
    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model_views, dictionary)
    top_5_words_comments = get_top_five_naive_bayes_words(naive_bayes_model_comments, dictionary)
    
    print('The top 5 most successful words are: ', top_5_words)
    print('Based on comments, the top 5 most successful words are: ', top_5_words_comments)
    #linear = LinearModel()
    #linear.fit(train_matrix,number_views)
    

if __name__ == "__main__":
    main()

