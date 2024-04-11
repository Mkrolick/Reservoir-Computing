import numpy as np




# Echo State Network class!
# Uses teacher forcing to align network output with desired output

class Resevoir:
    def __init__(self, input_size = 1, output_size = 1, resevoir_size = 20, spectral_radius = 0.95, sparsity = 0.1, input_function = "linear", seed = 42):
        
        # Set Seed
        np.random.seed(seed)
        self.seed = seed

        # Initialize Basic Parameters
        self.input_size = input_size
        self.resevoir_size = resevoir_size
        self.output_size = output_size

        # Initialize Input
        self.input = np.zeros((input_size, 1))

        # Initialize Resevoir
        # Generates weights from -1 to 1 uniformly for bias
        # Generates weight matrix from -1 to 1 uniformly for weight matrix

        self.input_weight_matrix = np.random.uniform(-1, 1, (resevoir_size, input_size)) # Size (N, k)
        self.weight_matrix = np.random.uniform(-1, 1, (resevoir_size, resevoir_size)) # Size (N, N)
        self.feedback_weight_matrix = np.random.uniform(-1, 1, (resevoir_size, output_size)) # Size (N, O)
        self.output_weight_matricies = np.array([np.random.uniform(-1, 1, (input_size + resevoir_size)) for i in range(output_size)]) # Size (O, k+N)
        
        # set sparsity of the matrix & set spectral radius (needed to help with stability)
        self.weight_matrix[np.random.rand(resevoir_size, resevoir_size) > sparsity] = 0 # for larger scale implementations use C++ and sparse matrices (hash map + set)
        
        try:
            self.weight_matrix *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.weight_matrix)))
        except (RuntimeError):
            print("Matrix is singular")
            return

        # set current state
        self.state = np.zeros((resevoir_size + input_size, 1)) # Size (N, 1)

        # set output
        self.output = np.zeros((output_size, 1)) # randomize for now
        self.output_coef = np.max(np.abs(np.linalg.eigvals(self.weight_matrix)))

        # set up resevoir states for backpropagation
        self.states = []
        self.outputs = []

        # Setup input function 
        self.input_function = input_function

    def predict(self, input, force=False, output_override = None):

        # Update Resevoir
        input = np.array(input).reshape((self.input_size, 1))

        # Output calculation is of the form
        # x(n+1) = tanh(W * x(n) + W_in * u(n+1) + W_fb * y(n)
        # where:
        # x(n+1) is the new state at time step x+1
        # x(n) is the current state
        # W is the weight matrix of the resevoir
        # W_in is the input weight matrix
        # u(n+1) is the input at time step n+1
        # W_fb is the feedback weight matrix
        # y(n) is the output at time step n

        if not force:
            temp_state = np.dot(self.weight_matrix, self.state[2:]) + np.dot(self.input_weight_matrix, input) #+ np.dot(self.feedback_weight_matrix, self.output)
        else:
            temp_state = np.dot(self.weight_matrix, self.state[2:]) + np.dot(self.input_weight_matrix, input) #+ np.dot(self.feedback_weight_matrix, output_override)
        
        # Apply Activation Function
        if self.input_function == "linear":
            self.state = temp_state
        elif self.input_function == "tanh":
            self.state = np.tanh(temp_state)
        elif self.input_function == "relu":
            self.state = np.maximum(0, temp_state)
        elif self.input_function == "sigmoid":
            self.state = 1/(1 + np.exp(-temp_state))
        else:
            print("Invalid Input Function")
            return 
            
        # add state to states
        # make state a column vector by suming values
        self.state = np.sum(self.state, axis = 1).reshape((self.resevoir_size, 1))

        # add input to state
        self.state = np.concatenate((input, self.state), axis = 0)

        self.states.append(self.state)

        # Calculate Output for each output matricies
        # y(n) = W_out * x(n) 

        self.output = np.zeros((self.output_size, 1))
        for index, matricies in enumerate(self.output_weight_matricies):
            self.output[index] = np.dot(matricies, self.state)
        
        #  add output to outputs
        self.outputs.append(self.output)

        # return squeezed output for clarity in reading
        return np.squeeze(self.output)

    def train(self, inputs, outputs):
        
        # Generate System States
        self.states = []
        self.outputs = []

        # Requires that output is one step ahead of input 
        # for a teaching signal
        for i in range(inputs.shape[0]):
            self.predict(inputs[i], force = True, output_override = outputs[i])


        # The desired output weights Wout are the linear regression weights of the desired outputs d(n) on the harvested extended states z(n)
        # A mathematically straightforward way to compute Wout is to invoke the pseudoinverse (denoted by ⋅†) of S : Wout = D†Z

        self.output_weight_matricies = np.dot(np.linalg.pinv(np.concatenate(self.states, axis = 1)).T, outputs).T

        # clear internal states
        resevoir.clear_states()

    def clear_states(self):

        # Clear states for new training
        self.states = []
        self.outputs = []

        # Copy output weight matricies
        output_matricies = self.output_weight_matricies

        # Reinitialize Resevoir
        self.__init__(input_size = self.input_size, output_size = self.output_size, resevoir_size = self.resevoir_size, input_function = self.input_function, seed=self.seed)

        # Set output weight matricies
        self.output_weight_matricies = output_matricies
        
if __name__ == "__main__":

    # Initialize Resevoir Parameters
    resevoir_size = 1000
    input_size = 2
    output_size = 3

    resevoir = Resevoir(input_size = input_size, output_size = output_size, resevoir_size = resevoir_size)

    # Set up training data
    X_train = np.array([[1,2],[4,2],[1,2],[4,2],[1,2],[4,2]])
    Y_pred = np.array([[3,6,9],[6,6,6],[3,6,9],[6,6,6],[3,6,9],[6,6,6]])

    # Train Resevoir
    resevoir.train(X_train, Y_pred)

    # It works!
    print("input: [1,2],",resevoir.predict(np.array([1,2])))
    print("input: [4,2],", resevoir.predict(np.array([4,2])))
    




        
