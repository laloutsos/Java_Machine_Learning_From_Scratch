import java.util.Random;

public class RegressionDataGenerator {


    /**
     * Generates a random dataset suitable for Least Squares regression.
     * @param nSamples Number of samples
     * @param nFeatures Number of features per sample
     * @param noise Standard deviation of Gaussian noise (0 for no noise)
     * @return Dataset object containing X, Y, and true weights
     */
    public static Dataset generateDataset(int nSamples, int nFeatures, double noise) {
        Random rand = new Random();

        double[][] X = new double[nSamples][nFeatures];
        double[] Y = new double[nSamples];
        double[] trueWeights = new double[nFeatures];

        // Generate random true weights
        for (int j = 0; j < nFeatures; j++) {
            trueWeights[j] = rand.nextDouble() * 10 - 5; // random weights between -5 and 5
        }

        // Generate feature matrix X and target vector Y
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = rand.nextDouble() * 10; // random feature values between 0 and 10
            }

            // Compute Y = X * trueWeights
            double yValue = 0;
            for (int j = 0; j < nFeatures; j++) {
                yValue += X[i][j] * trueWeights[j];
            }

            // Add optional Gaussian noise
            yValue += rand.nextGaussian() * noise;

            Y[i] = yValue;
        }

        return new Dataset(X, Y, trueWeights);
    }

    public static Dataset generateNewData(int nSamples, double[] trueWeights, double noise) {
        Random rand = new Random();
        int nFeatures = trueWeights.length;

        double[][] X = new double[nSamples][nFeatures];
        double[] Y = new double[nSamples];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = rand.nextDouble() * 10; // same feature range
            }

            double yValue = 0;
            for (int j = 0; j < nFeatures; j++) {
                yValue += X[i][j] * trueWeights[j];
            }

            yValue += rand.nextGaussian() * noise; // optional noise
            Y[i] = yValue;
        }

        return new Dataset(X, Y, trueWeights);
    }

    // Example usage
    public static void main(String[] args) {
        // Generate dataset with 100 samples, 3 features, small noise
        Dataset dataset = generateDataset(100, 3, 0.1);

        LeastSquares ls = new LeastSquares();
        double[] estimatedWeights = ls.LeastSquaresEstimator(dataset.X, dataset.Y);

        System.out.println("True weights:");
        for (double w : dataset.trueWeights) System.out.print(w + " ");
        System.out.println("\nEstimated weights:");
        for (double w : estimatedWeights) System.out.print(w + " ");
    }
}
