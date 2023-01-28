
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))
                
    def one_hot(self,y):
        class_labels = [i for i in range(10)]
        one_hot = np.eye(self.num_classes)[np.vectorize(lambda c: class_labels[c])(y).reshape(-1)]
        for i in range(len(y)):
            one_hot[i] = one_hot[i] * y[i]
        return one_hot
    
    # def softmax(self, scores):
    #     f = np.exp(scores - np.max(scores))  # shift values
    #     return f / np.sum(f)

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################

        # Z = X * W
        Z = np.matmul(X,self.weights['W1'])
        A = self.ReLU(Z)
        p = self.softmax(A)
        
        accuracy = self.compute_accuracy(p,y)
        loss = self.cross_entropy_loss(p,y)
        
        


        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        
        error = p
        error[range(len(y)),y] -= 1
        
        
        dA = self.ReLU_dev(A) # derivative of ReLU activation with respect to Z
        dZ = 1/len(y)* error * dA 
        dW = np.dot(X.T, dZ)
        
        self.gradients['W1'] += dW # update gradients
        
        

        return loss, accuracy

