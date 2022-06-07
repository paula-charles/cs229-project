import numpy as np

class LinearModel(object):
    def __init__(self, step_size=1e-3, max_iter=1000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, uses the zero vector.
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
        The polynomial map has powers from 0 to k
        Output is a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """

        n_examples = len(X)
        augmented_X = np.zeros(shape=(n_examples,k+1))
        for i in range(len(X)):
            for j in range(k+1):
                augmented_X[i,j] = X[i,1]**j
        return augmented_X    

    
    def predict(self, X):
        """
        Makes a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        return np.dot(self.theta,X.T)

def fit_naive_bayes(matrix, labels):
    """
    This function fits a Naive Bayes model given a training matrix and labels.
     It returns the state of that model.

   Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

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

def predict_naive_bayes(model, matrix):
    """
    This function predicts on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """

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

def get_top_words_naive_bayes(nb, model, dictionary):
    """
    This function computes the top words that appear in the most viewed titles.
    The argument nb states how many words we are returning.
    Return the words in sorted form, with the most popular word first.
    """

    avg_perf_word = model[1]
    indices = []
    for k in range(5):
        ind = np.argmax(avg_perf_word)
        indices.append(ind)
        avg_perf_word[ind] = 0

    return [list(dictionary.keys())[indices[k]] for k in range(nb)]
