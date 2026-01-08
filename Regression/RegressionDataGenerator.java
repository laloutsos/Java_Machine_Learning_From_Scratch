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

    public static Dataset generateSinDataset(int nSamples, double noise) {
        Random rand = new Random();
        double[][] X = new double[nSamples][1];
        double[] Y = new double[nSamples];

        for (int i = 0; i < nSamples; i++) {
            X[i][0] = rand.nextDouble() * 10; // X in [0,10]
            Y[i] = Math.sin(X[i][0]) + rand.nextGaussian() * noise;
        }

        return new Dataset(X, Y, null);
    }

    /**
     * Generate a dataset with Y = x + x^2 + ... + x^degree + Gaussian noise
     */
    public static Dataset generatePolynomialDataset(int nSamples, double noise, int degree) {
        Random rand = new Random();
        double[][] X = new double[nSamples][1];
        double[] Y = new double[nSamples];

        for (int i = 0; i < nSamples; i++) {
            X[i][0] = rand.nextDouble() * 5; // X in [0,5]
            double x = X[i][0];
            double yValue = 0;

            // Sum powers of x up to degree
            for (int d = 1; d <= degree; d++) {
                yValue += Math.pow(x, d);
            }

            // Add Gaussian noise
            yValue += rand.nextGaussian() * noise;
            Y[i] = yValue;
        }

        return new Dataset(X, Y, null);
    }


    /**
     * Generate a dataset with Y = exp(-0.1X) + cos(0.5X) + Gaussian noise
     */
    public static Dataset generateExpCosDataset(int nSamples, double noise) {
        Random rand = new Random();
        double[][] X = new double[nSamples][1];
        double[] Y = new double[nSamples];

        for (int i = 0; i < nSamples; i++) {
            X[i][0] = rand.nextDouble() * 20; // X in [0,20]
            double x = X[i][0];
            Y[i] = Math.exp(-0.1*x) + Math.cos(0.5*x) + rand.nextGaussian() * noise;
        }

        return new Dataset(X, Y, null);
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
