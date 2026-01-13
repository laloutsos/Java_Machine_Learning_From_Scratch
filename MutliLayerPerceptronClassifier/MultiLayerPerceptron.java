import java.util.function.DoubleUnaryOperator;
import java.util.*;


public class MultiLayerPerceptron {
    private Layer[] layers;
    private int inputSize;
    private double accuracy;

    // Initialize the MLP with a constructor
    public MultiLayerPerceptron(int inputSize, int[] H, int outputSize,DoubleUnaryOperator[] activation,
                                DoubleUnaryOperator[] activationDerivative){
        this.inputSize = inputSize;
        this.accuracy = accuracy;

        layers = new Layer[H.length+1];
        int previousSize = inputSize;
        for(int i = 0; i<H.length; i++){
            layers[i] = new Layer(H[i], previousSize,activation[i], activationDerivative[i]);
            previousSize = H[i];
        }

        layers[H.length] = new Layer(outputSize, previousSize, activation[activation.length-1], activationDerivative[activation.length-1]);
    }

    // forward pass between the layers
    public double[] forwardPass(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.layerForward(current);
        }
        return current;
    }

    // backward pass between the layers
    public void backwardPass(double[] dLoss_dOutput) {
        double[] gradient = dLoss_dOutput;
        for (int i = layers.length - 1; i >= 0; i--) {
            gradient = layers[i].layerBackward(gradient);
        }
    }


    // Gradient Descent
    public void train(List<double[]> X, List<double[]> T, int maxEpochs, int batchSize, double learningRate, double errorThreshold) {
        int N = X.size();
        double previousEpochError = Double.POSITIVE_INFINITY;
        int epoch = 0;

        while (true) {
            double currentEpochError = 0.0;

            for (int batchStart = 0; batchStart < N; batchStart += batchSize) {
                // Reset the gradients!
                for (Layer layer : layers)
                    layer.zeroGradients();
                // Finds the end even if theres assymmetry
                int end = Math.min(batchStart + batchSize, N);

                for (int i = batchStart; i < end; i++) {
                    double[] x = X.get(i);
                    double[] t = T.get(i);

                    double[] y_hat = forwardPass(x);
                    double[] dLoss_dOutput = new double[y_hat.length];

                    for (int j = 0; j < y_hat.length; j++) {
                        // compute the error
                        double diff = y_hat[j] - t[j];
                        currentEpochError += diff * diff;
                        dLoss_dOutput[j] = 2 * diff;
                    }
                    // backward pass the error through the network
                    backwardPass(dLoss_dOutput);
                }

                for (Layer layer : layers) {
                    // we want to normalise the learning rate with the batch size
                    layer.layerUpdate(learningRate / (end - batchStart));
                }
            }

            double meanSquaredError = currentEpochError / N;
            System.out.printf("Epoch %d: Training Error (MSE) = %.6f%n", epoch, meanSquaredError);

            if (epoch >= 799) {
                double errorDifference = Math.abs(previousEpochError - meanSquaredError);
                if (errorDifference < errorThreshold) break;
            }

            if (epoch >= maxEpochs - 1) break;

            previousEpochError = meanSquaredError;
            epoch++;
        }
    }

    public List<String> test(List<double[]> X_test, List<double[]> Y_test, MultiLayerPerceptron mlp) {
        List<String> results = new ArrayList<>();
        int correct = 0;

        for (int i = 0; i < X_test.size(); i++) {
            double[] x = X_test.get(i);
            double[] output = mlp.forwardPass(x);

            // --- Find predicted class ---
            int predictedClass = 0;
            double maxVal = output[0];
            for (int j = 1; j < output.length; j++) {
                if (output[j] > maxVal) {
                    maxVal = output[j];
                    predictedClass = j;
                }
            }

            // --- Find actual class from one-hot ---
            double[] trueLabel = Y_test.get(i);
            int actualClass = 0;
            for (int j = 1; j < trueLabel.length; j++) {
                if (trueLabel[j] > trueLabel[actualClass]) {
                    actualClass = j;
                }
            }

            // --- Count accuracy ---
            if (predictedClass == actualClass) {
                correct++;
            }

            // --- Create readable result line ---
            StringBuilder sb = new StringBuilder();
            sb.append("Input: ").append(Arrays.toString(x));
            sb.append(" | Predicted: ").append(predictedClass);
            sb.append(" | Actual: ").append(actualClass);

            results.add(sb.toString());
        }

        double accuracyy = correct / (double) X_test.size();
        accuracy = accuracyy * 100;
        System.out.printf("Accuracy on test set: %.2f%%%n", accuracy);

        return results;
    }

    public double getAccuracy(){
        return this.accuracy;
    }

}
