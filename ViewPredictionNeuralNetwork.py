import numpy as np
import matplotlib.pyplot as plt
import argparse
import Naive_Bayes_approach as NBa
import pandas as pd

def ReLU(x):
    """
    Compute the rectified linear function of the input.
    
    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the ReLU results
    """
    return x * (x >= 0)
    
def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1/(1+np.exp(-x))
    
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    Weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    W1 = np.random.randn(input_size,num_hidden)
    b1 = np.zeros(num_hidden)
    W2 = np.random.randn(num_hidden,num_output)
    b2 = np.zeros(num_output)
    
    dictionary = {"W1":W1,"W2":W2,"b1":b1,"b2":b2}
    return dictionary
    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    nExamples= len(labels)
    nBits = len(data[0])
    nHidden = len(params['b1'])
    nFeatures = len(labels[0])
   
    #B_1 = np.tile(params['b1'],(nExamples,1))
    #Z_1 = np.matmul(data,params['W1'])+params['b1']#B_1
    Z_1 = data.dot(params['W1'])+params['b1']
    A_1 = sigmoid(Z_1)
    #B_2 = np.tile(params['b2'],(nExamples,1))
    #Z_2 = np.matmul(A_1,params['W2']) +params['b2']#B_2
    Z_2 = A_1.dot(params['W2'])+params['b2']
    complete_prediction = ReLU(Z_2)
    print(np.exp(complete_prediction[:10]))

    #print(complete_prediction[:10])
    #print(labels[0:10])

    avg_loss= 1/nExamples*np.sum(abs(np.exp(labels)-np.exp(complete_prediction)))
    
    #print("Average_CE_loss is ",CE_loss/n)
    #print(np.shape(A_1),np.shape(complete_prediction))
    return (A_1,complete_prediction,avg_loss)
    
    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    nExamples = len(labels) #Batch-size or number of examples
    nBits = len(data[0])
    nHidden = len(params['b1'])
    nFeatures = len(labels[0])
    a_1,y_hat,average_loss = forward_prop_func(data,labels,params)
        
    #Gradz2 =np.zeros((B,nFeatures))
    Gradz2 = 1/nExamples*(y_hat>=0)*(y_hat-labels)
    #Gradz2 = 1/B*(y_hat-labels)
    Grada1 = np.matmul(Gradz2,np.transpose(params["W2"]))
    Gradz1=Grada1*a_1*(1-a_1) #true for the sigmoid

    gradW2 = np.matmul(np.transpose(a_1),Gradz2)
    gradW1 = np.matmul(np.transpose(data),Gradz1)
    gradb2 = np.sum(Gradz2,axis=0) 
    gradb1 = np.sum(Gradz1,axis=0)
    #print(np.shape(gradb1),np.shape(gradb2),np.shape(gradW1),np.shape(gradW2))
    
    
    dico = {"W1":gradW1,"W2":gradW2,"b1":gradb1,"b2":gradb2}
    return dico
    
    # *** END CODE HERE ***


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    B = len(labels) #Batch-size or number of examples
    nBits = len(data[0])
    nHidden = len(params['b1'])
    nFeatures = len(labels[0])
    a_1,y_hat,average_loss = forward_prop_func(data,labels,params)
    
    def sigmoid_derivative(x):
        return np.exp(-x)/(1+np.exp(-x))**2
        
    Gradz2 =np.zeros((B,nFeatures))
    Gradz2 = 1/B*(y_hat>=0)*(y_hat-labels)
    #Gradz2 = 1/B*(y_hat-labels)
    Grada1 = np.matmul(Gradz2,np.transpose(params["W2"]))
    Gradz1=Grada1*a_1*(1-a_1) #true for the sigmoid

    gradW2 = np.matmul(np.transpose(a_1),Gradz2) + 2*reg*params["W2"]
    gradW1 = np.matmul(np.transpose(data),Gradz1) + 2*reg*params["W1"]
    gradb2 = sum(Gradz2)
    gradb1 = sum(Gradz1)
    
    dico = {"W1":gradW1,"W2":gradW2,"b1":gradb1,"b2":gradb2}
    return dico
    
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    print(forward_prop(train_data,train_labels,params)[1][0:20])
    nExamples = len(train_data)
    total_batch_number = int(nExamples/batch_size)
    for i in range(total_batch_number): #batch number
        gradient_dico = backward_prop_func(train_data[batch_size*i:min(nExamples,batch_size*(i+1))],train_labels[batch_size*i:min(nExamples,batch_size*(i+1))],params,forward_prop_func)
    
        params["W2"] -= learning_rate*gradient_dico["W2"]
        params["W1"] -= learning_rate*gradient_dico["W1"]
        params["b1"] -= learning_rate*gradient_dico["b1"]
        params["b2"] -= learning_rate*gradient_dico["b2"]
    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=20, learning_rate=0.001, num_epochs=30, batch_size=100):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 1)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        
        gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))
    
    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    nExamples = len(labels)
    accuracy = np.average(abs(output-labels)/output)
    return accuracy





def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=20, learning_rate=0.001, num_epochs=num_epochs, batch_size=100
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)
    
    args = parser.parse_args()
    np.random.seed(100)
    
    df = pd.read_csv('ted_main.csv', delimiter=',')
    tuples = [list(x) for x in df.values]
    
    n_examples_total = len(tuples)
    n_test = int(n_examples_total*0.1)
    n_val = int(n_examples_total*0.3)
    
    #train_names = [list(datapoint[7].rpartition(": "))[-1] for datapoint in tuples]
    #train_names = [datapoint[-3] for datapoint in tuples]
    train_description = [datapoint[1] for datapoint in tuples]
    
    train_names = [datapoint[1] for datapoint in tuples]
    train_names = [datapoint[13] for datapoint in tuples]
    
    dictionary = NBa.create_dictionary(train_names)
    print('Size of dictionary: ', len(dictionary))
    train_matrix = NBa.transform_text(train_names, dictionary)
    train_number_views = np.array([[np.log(datapoint[-1])] for datapoint in tuples])
    """
    training_data = tuples[:n_examples]
    validation_data = tuples[n_examples:n_val]
    test_data = tuples[n_val:]
    

    train_number_comments = np.array([datapoint[0] for datapoint in training_data])
    
    
  
    val_names = [list(datapoint[7].rpartition(": "))[-1] for datapoint in validation_data]
    val_matrix = NBa.transform_text(val_names, dictionary)
    val_number_comments = [datapoint[0] for datapoint in validation_data]
    val_number_views = [datapoint[-1] for datapoint in validation_data]
    
    val_matrix = transform_text(val_names, dictionary)
    """
    
    train_data, train_labels =  train_matrix,train_number_views
    
    
    p = np.random.permutation(n_examples_total)
    train_data = train_data[p,:]
    train_labels = train_labels[p]
    
    test_data = train_data[0:n_test,:]
    test_labels = train_labels[0:n_test]
    dev_data = train_data[n_test:n_val,:]
    dev_labels = train_labels[n_test:n_val]
    train_data = train_data[n_val:,:]
    train_labels = train_labels[n_val:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    #test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    #test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)
        
    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
