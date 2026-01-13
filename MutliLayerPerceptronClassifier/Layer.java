import java.util.function.DoubleUnaryOperator;


public class Layer {
    private Neuron[] neurons;
    private int inputSize;

    public Layer(int numberOfNeurons, int inputSize,
                 DoubleUnaryOperator activation,
                 DoubleUnaryOperator activationDerivative) {
        neurons = new Neuron[numberOfNeurons];
        this.inputSize = inputSize;
        for (int i = 0; i < numberOfNeurons; i++) {
            neurons[i] = new Neuron(inputSize, activation, activationDerivative);
        }
    }

    public double[] layerForward(double[] input){
        // forward between the neurons of each layer
        double[] output = new double[neurons.length];
        for(int i=0; i<neurons.length; i++){
            output[i] = neurons[i].forward(input);

        }
        return output;
    }

    public double[] layerBackward(double[] dLoss_dOutput) {
        double[] prevLayerGradient = new double[inputSize];
        // backward between the neuorons of every layer
        for (int i = 0; i < neurons.length; i++) {
            double[] neuronInputGradient = neurons[i].backward(dLoss_dOutput[i]);
            for (int j = 0; j < neuronInputGradient.length; j++) {
                prevLayerGradient[j] += neuronInputGradient[j];
            }
        }
        return prevLayerGradient;
    }

    // Update weights for all neurons
    public void layerUpdate(double learningRate) {
        for (Neuron n : neurons) {
            n.update(learningRate);
        }
    }
    // we want to reset gradients after each batch!!
    public void zeroGradients() {
        for (Neuron n : neurons)
            n.zeroGradients();
    }



}
