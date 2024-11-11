import numpy as np
from  optimizer import optimizer

class NeuralNetwork:
    def __init__(self,layers,optimizer_type,learning_rate,beta1,beta2,epsilon):

        self.weights = {} # dictionary to store weight matrix
        self.biases = {} # dictionary to store layer biases 
        self.layers = layers

        

        #loop to initialize the initial weight matrix
        for i in range(0,len(layers)-1):
            self.weights[f'layer{i}_to_layer{i+1}_weights'] = np.random.rand(layers[i],layers[i+1])
        
        #loop to intialize the initial bias function 
        for j in range(1,len(layers)):
            self.biases[f'layer{j}_bias'] = np.zeros(layers[j]) #setting the biases to 0 initially

        self.optimizer = optimizer(optimizer_type, learning_rate, beta1, beta2, epsilon,layers)


        

    def get_model_params(self):
        return self.weights,self.biases # returns weights and biases of the model

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        sig = self.sigmoid(x)
        return sig*(1-sig)
    
    def ReLU(self,x):
        return np.maximum(0,x)
    
    def ReLU_derivative(self,x):
        return np.where(x>0,1,0)
    

    def forward_pass(self,X):

        self.z_values = {} #pre-activation values
        self.layer_output = {} #post activation values 
        layer_operations = {}

        self.layer_output['layer0_output'] = X

        #forward propogation loop excluding the output layer calculations (cause of different activation functions)
        for i in range(1,len(self.biases),1): 
            weight_key = f'layer{i-1}_to_layer{i}_weights'
            bias_key = f'layer{i}_bias'
            z_key = f'z{i}'
            h_key = f'layer{i}_output'

            self.z_values[z_key] = np.dot(self.layer_output[f'layer{i-1}_output'],self.weights[weight_key]) + self.biases[bias_key] 
            self.layer_output[h_key] = self.ReLU(self.z_values[z_key])
            # print(self.layer_output[h_key])

        # calculating the output layer output using the sigmoid activation 
        self.output_layer_index = len(self.biases) 
        output_layer_index = len(self.biases) 
        weight_key = f'layer{output_layer_index-1}_to_layer{output_layer_index}_weights'
        bias_key = f'layer{output_layer_index}_bias'
        z_key = f'z{output_layer_index}'
        h_key = f'layer{output_layer_index}_output'

        self.z_values[z_key] = np.dot(self.layer_output[f'layer{output_layer_index-1}_output'], self.weights[weight_key]) + self.biases[bias_key]
        self.layer_output[h_key] = self.sigmoid(self.z_values[z_key])
        # print(self.layer_output[f'layer{output_layer_index}_output'])

        self.y_hat = self.layer_output[h_key] # final predicted output 
        


    def backward_pass(self,X,y,learning_rate):

        # creating dictionary for storing the gradients and errors
        self.gradients = {}
        self.errors = {}

        #calculating the error and gradients for the output layer function seperately (cause it has sigmoid derivative)
        self.net_error = y - self.y_hat 
        self.errors[f'layer{self.output_layer_index}_error'] = self.net_error*self.sigmoid_derivative(self.z_values[f'z{self.output_layer_index}'])
        self.gradients[f'layer{self.output_layer_index-1}_to_layer{self.output_layer_index}_weights_gradient'] = np.dot(self.layer_output[f'layer{self.output_layer_index-1}_output'].T,self.errors[f'layer{self.output_layer_index}_error'])*learning_rate


        for i in range(self.output_layer_index-1,0,-1):
            error_key = f'layer{i}_error'
            self.errors[error_key] = np.dot(self.errors[f'layer{i+1}_error'],self.weights[f'layer{i}_to_layer{i+1}_weights'].T)*self.ReLU_derivative(self.z_values[f'z{i}'])

        for j in range(self.output_layer_index-1,0,-1):
            gradient_key = f'layer{j-1}_to_layer{j}_weights_gradient'
            h_key = f'layer{j-1}_output' 
            error_key = f'layer{j}_error'
            self.gradients[gradient_key] = np.dot(self.layer_output[h_key].T,self.errors[error_key])
    


    # def adagrad(self,epsilon):




    def parameter_updates(self,learning_rate):

        #loop for parameter updates 
        for i in range(1,len(self.biases)):
            weight_key = f'layer{i-1}_to_layer{i}_weights'
            gradient_key = f'layer{i-1}_to_layer{i}_weights_gradient'
            error_key = f'layer{i}_error' 
            bias_key = f'layer{i}_bias'



            self.weights[weight_key] = self.weights[weight_key] - self.gradients[gradient_key]
            self.biases[bias_key] = self.biases[bias_key] - np.sum(self.errors[error_key],axis=0)*learning_rate

    
    def train(self, X, y, epochs, batch_size=None,learning_rate=0.01):
        for epoch in range(epochs):
            if batch_size:
                # Mini-batch gradient descent
                indices = np.random.permutation(X.shape[0])
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for i in range(0, X.shape[0], batch_size):
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]
                    y_batch_one_hot = np.eye(np.max(y) + 1)[y_batch]

                    self.forward_pass(X_batch)
                    self.backward_pass(X_batch, y_batch_one_hot,learning_rate=0.01)
                    self.optimizer.update_parameters(self, self.gradients)  # Using optimizer to update parameters
            else:
                # Full-batch gradient descent
                y_one_hot = np.eye(np.max(y) + 1)[y]
                self.forward_pass(X)
                self.backward_pass(X, y_one_hot)
                self.optimizer.update_parameters(self, self.gradients)

            if epoch % 100 == 0:
                loss = self.compute_loss(y_one_hot)
                print(f'Epoch: {epoch}, Loss: {loss}')



    def compute_loss(self, y):
    # Compute cross-entropy loss for multi-class classification
        m = y.shape[0]  # Number of samples
        y_true_labels = np.argmax(y, axis=1)  # Get the true labels from one-hot encoding
        log_likelihood = -np.log(self.y_hat[np.arange(m), y_true_labels])
        loss = np.sum(log_likelihood) / m
        return loss

            
    def model_architecture(self):
        num_layers = len(self.layers)  # Total number of layers, including input, hidden, and output

        for i in range(num_layers):
            if i == 0:
                # Input layer
                print(f'Layer {i}: Input layer with {self.layers[i]} neurons')

            elif i == num_layers - 1:
                # Output layer
                print(f'Layer {i}: Output layer with {self.layers[i]} neurons (Sigmoid activation)')
            else:
                # Hidden layers
                print(f'Layer {i}: Hidden layer with {self.layers[i]} neurons (ReLU activation)')
        
            # Print the size of weights between this layer and the next
            if i < num_layers - 1:
                print(f'   Weights shape: {self.weights[f"layer{i}_to_layer{i+1}_weights"].shape}')
                print(f'   Biases shape: {self.biases[f"layer{i+1}_bias"].shape}')
        
        print("End of Architecture")


        
            
        

