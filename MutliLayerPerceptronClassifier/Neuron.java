import java.util.function.DoubleUnaryOperator;

public class Neuron {
    private double[] weights; // Array of Weights for each input
    private double bias; // Bias term

    private double[] input; // Input vector
    private double sum; // Weighted sum before activation
    private double output; // output after activation

    private double[] weightGradient; // Gradients for each weight
    private double biasGradient; // Gradient for bias
    private double[] inputGradient; // Gradient to pass back to previous layer


    private java.util.function.DoubleUnaryOperator activation; // activation function
    private java.util.function.DoubleUnaryOperator activationDerivative; // Derivative of activation function

    public static final DoubleUnaryOperator SIGMOID = x -> 1 / (1 + Math.exp(-x));
    public static final DoubleUnaryOperator SIGMOID_DERIVATIVE = x -> {
        double s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
    };

    // ReLU
    public static final DoubleUnaryOperator RELU = x -> Math.max(0, x);
    public static final DoubleUnaryOperator RELU_DERIVATIVE = x -> x > 0 ? 1 : 0;

    // Tanh
    public static final DoubleUnaryOperator TANH = Math::tanh;
    public static final DoubleUnaryOperator TANH_DERIVATIVE = x -> 1 - Math.pow(Math.tanh(x), 2);

    public Neuron(int inputSize, java.util.function.DoubleUnaryOperator activation
    , java.util.function.DoubleUnaryOperator activationDerivative){

        this.weights = new double[inputSize]; // Initialize weights array
        this.weightGradient = new double[inputSize]; // Initialize gradient array for weights
        this.inputGradient = new double[inputSize]; // Initialize gradient array for inputs
        this.activation = activation; //Set activation function
        this.activationDerivative = activationDerivative; // Set derivative of activation function
        // Randomly initialize weights and bias
        for (int i = 0; i<inputSize; i++){
            this.weights[i] = Math.random() - 0.5;
        }

        this.bias = Math.random() - 0.5;
    }

    public double forward(double[] input){
        this.input = input;
        sum = 0; // Reset the weighted sum
        for(int i=0; i<weights.length; i++){
            sum+= weights[i] * input[i];

        }
        sum+=bias;
        // Apply to the sum the activation function and return it

        output = activation.applyAsDouble(sum);
        return output;
    }

    // Backward pass for gradients

    public double[] backward(double errorTerm) {

        // δ = g'(u) * errorTerm
        double delta = activationDerivative.applyAsDouble(sum) * errorTerm;

        // 2. weight derivatives
        for (int i = 0; i < weights.length; i++) {
            weightGradient[i] += delta * input[i]; // if batch size = 1 -> weightGradient[i] = delta * input[i]; -> serial update
            inputGradient[i]  = delta * weights[i];
        }

        // 3. bias derivative
        biasGradient += delta;

        // 4. return gradient backwards
        return inputGradient;
    }

    public void update(double learningRate) { // Update weights and bias using gradients
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * weightGradient[i]; // Gradient descent step
        }
        bias -= learningRate * biasGradient; // Update bias
    }

    public double getOutput() { // Get output of neuron
        return output;
    }

    public double[] getWeights() { // Get current weights
        return weights;
    }

    public double getBias() { // Get current bias
        return bias;
    }

    // Reset the gradients 

    public void zeroGradients() {
        for (int i = 0; i < weightGradient.length; i++)
            weightGradient[i] = 0;
        biasGradient = 0;
    }

}
