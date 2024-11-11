import numpy as np 

class optimizer:
    def __init__(self,optimizer_type='sgd',learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-08):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon 
        self.v = {}
        self.s = {}
        self.t = 0



    def update_parameters(self, neural_net, gradients):
        if self.optimizer_type == 'sgd':
            self.sgd_update(neural_net, gradients)
        elif self.optimizer_type == 'momentum':
            self.momentum_update(neural_net, gradients)
        elif self.optimizer_type == 'adagrad':
            self.adagrad_update(neural_net, gradients)
        elif self.optimizer_type == 'rmsprop':
            self.rmsprop_update(neural_net, gradients)
        elif self.optimizer_type == 'adam':
            self.adam_update(neural_net, gradients)


    def sgd_update(self,neural_net,gradients):
        for key in gradients:
            neural_net.weights[key] -= self.learning_rate*gradients[key]

    
    def momentum_update(self,neural_net,  gradients):
        for key in gradients:
            if key not in self.v:
                self.v[key] = np.zeros_like(gradients[key])

            self.v[key] = self.bet1*self.v[key]  + (1-self.beta1)*gradients[key]
            neural_net.weights -= self.learning_rate*self.v[key]


    def adagrad_update(self, neural_net, gradients):
        for key in gradients:
            if key not in self.s:
                self.s[key] = np.zeros_like(gradients[key])
            self.s[key] += gradients[key] ** 2
            neural_net.weights[key] -= self.learning_rate * gradients[key] / (np.sqrt(self.s[key]) + self.epsilon)


    def rmsprop_update(self, neural_net, gradients):
        for key in gradients:
            if key not in self.s:
                self.s[key] = np.zeros_like(gradients[key])
            self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (gradients[key] ** 2)
            neural_net.weights[key] -= self.learning_rate * gradients[key] / (np.sqrt(self.s[key]) + self.epsilon)


    def adam_update(self, neural_net, gradients):
        self.t += 1
        for key in gradients:
            if key not in self.v:
                self.v[key] = np.zeros_like(gradients[key])
            if key not in self.s:
                self.s[key] = np.zeros_like(gradients[key])

            self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * gradients[key]
            self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (gradients[key] ** 2)

            v_corrected = self.v[key] / (1 - self.beta1 ** self.t)
            s_corrected = self.s[key] / (1 - self.beta2 ** self.t)

            neural_net.weights[key] -= self.learning_rate * v_corrected / (np.sqrt(s_corrected) + self.epsilon)

