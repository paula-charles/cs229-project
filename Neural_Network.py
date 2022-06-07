import numpy as np
import matplotlib.pyplot as plt
import argparse

def ReLU(x):
    """
    Computes the ReLU function
    """
    return x * (x >= 0)
    
def sigmoid(x):
    """
    Compute the sigmoid function.
    """
    return 1/(1+np.exp(-x))

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function returns a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    The four parameters for this model are:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output
    """
    np.random.seed(1)
    W1 = np.random.randn(input_size,num_hidden)
    b1 = np.zeros(num_hidden)
    W2 = np.random.randn(num_hidden,num_output)
    b2 = np.zeros(num_output)
    
    dictionary = {"W1":W1,"W2":W2,"b1":b1,"b2":b2}
    return dictionary

def forward_prop(data, labels, params, log_transfo):
    """
    Implement the forward layer given the data, labels, and params.
    """
    nExamples= len(labels)
    nBits = len(data[0])
    nHidden = len(params['b1'])
    nFeatures = len(labels[0])

    Z_1 = data.dot(params['W1'])+params['b1']
    A_1 = sigmoid(Z_1)
    Z_2 = A_1.dot(params['W2'])+params['b2']
    complete_prediction = ReLU(Z_2)
    #print(np.exp(complete_prediction[:10]))

    if log_transfo:
        avg_loss= 1/nExamples*np.sum(abs(np.exp(labels)-np.exp(complete_prediction)))
    else:
        avg_loss = 1 / nExamples * np.sum(abs(labels - complete_prediction))
    
    #print("Average_CE_loss is ",CE_loss/n)
    #print(np.shape(A_1),np.shape(complete_prediction))
    return (A_1,complete_prediction,avg_loss)


def backward_prop(data, labels, params, forward_prop_func, log_transfo):
    """
    Implement the backward propagation gradient computation step for a neural network
    """
    nExamples = len(labels) #Batch-size or number of examples
    nBits = len(data[0])
    nHidden = len(params['b1'])
    nFeatures = len(labels[0])
    a_1,y_hat,average_loss = forward_prop_func(data,labels,params, log_transfo)
        
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


def backward_prop_regularized(data, labels, params, forward_prop_func, log_transfo, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    """
    B = len(labels) #Batch-size or number of examples
    nBits = len(data[0])
    nHidden = len(params['b1'])
    nFeatures = len(labels[0])
    a_1,y_hat,average_loss = forward_prop_func(data,labels,params, log_transfo)
    
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

def gradient_descent_epoch(train_data, train_labels,
                           learning_rate, batch_size, params, forward_prop_func, backward_prop_func, log_transfo):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.
    """

    #print(forward_prop(train_data,train_labels,params)[1][0:20])
    nExamples = len(train_data)
    total_batch_number = int(nExamples/batch_size)
    for i in range(total_batch_number): #batch number
        gradient_dico = backward_prop_func(train_data[batch_size*i:min(nExamples,batch_size*(i+1))],
                                           train_labels[batch_size*i:min(nExamples,batch_size*(i+1))],
                                           params,forward_prop_func, log_transfo)
    
        params["W2"] -= learning_rate*gradient_dico["W2"]
        params["W1"] -= learning_rate*gradient_dico["W1"]
        params["b1"] -= learning_rate*gradient_dico["b1"]
        params["b2"] -= learning_rate*gradient_dico["b2"]

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func, log_transfo,
    num_hidden=20, learning_rate=0.001, num_epochs=30, batch_size=100):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 1)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        
        gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size,
                               params, forward_prop_func, backward_prop_func, log_transfo)

        h, output, cost = forward_prop_func(train_data, train_labels, params, log_transfo)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params, log_transfo)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))
    
    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params, log_transfo):
    h, output, cost = forward_prop(data, labels, params, log_transfo)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    nExamples = len(labels)
    accuracy = np.average(2*abs(output-labels)/(output+labels))
    return accuracy

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, backward_prop, log_transfo, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func, log_transfo,
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

    accuracy = nn_test(all_data['test'], all_labels['test'], params, log_transfo)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

