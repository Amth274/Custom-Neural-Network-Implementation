import  numpy as np


class NeuralNetwork:
    def __init__(self,layers,activations):
        self.weights = {}
        self.biases = {}
        self.layers = layers
        self.activations = activations
        self.output_layer_index = len(layers)-1

        for i in range(len(layers)):
            self.weights_key = f'layer{i}_to_layer{i+1}_weights'
            self.bias_key = f'layer{i+1}_bias'
            

        


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def activation_function(self,activation_name):

        self.activation_dict = {'relu':self.ReLU,'sigmoid':self.sigmoid}
        return self.activation_dict.get(activation_name) 
    
    def activation_derivative(self,activation_name):
        self.activation_derivative_dict = {'relu':self.ReLU_derivative,'sigmoid':self.sigmoid_derivative}
        return self.activation_derivative_dict.get(activation_name)

    def forward_pass(self,X):

        self.z = {} #post activation layer output 
        self.h = {} #pre activation layer output
        
        self.z['layer0_output'] = X 

        #loop for forward propogation 
        for i in range(1,len(self.layers)):
            h_key = f'h{i}'
            z_key = f'layer{i}_output'
            weights_key = f'layer{i-1}_to_layer{i}_weights'

            self.h[h_key] = np.dot(self.z[f'layer{i-1}_output'],self.weights[weights_key]) + self.biases[f'layer{i}_bias']
            activation_func = self.activation_function(self.activations[i-1]) #calls the corresponding activation function selected 
            self.z[z_key] = activation_func(self.h[h_key]) 

        self.y_hat = self.z[f'layer{len(self.layers)-1}_output'] 

    
    def backward_pass(self,X,y,learning_rate):

        self.error = y - self.y_hat

        self.errors = {}
        self.gradients = {}
        activation_derivative = self.activation_derivative(self.activations[i-1])
        self.errors[f'layer{self.output_layer_index}_error'] = self.error*activation_derivative(self.h[f'h{self.output_layer_index}'])

        for i in range(len(self.output_layer_index),0,-1):
            error_key = f'layer{i}_error'

            self.errors(error_key) = np.dot(self.errors[f'layer{i+1}_error'],)      

        

