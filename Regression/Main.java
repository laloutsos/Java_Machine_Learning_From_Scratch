import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // 1️ Ask user for dataset parameters
        System.out.print("Number of training samples: ");
        int nSamples = sc.nextInt();

        System.out.print("Number of features: ");
        int nFeatures = sc.nextInt();

        System.out.print("Noise standard deviation (e.g., 0.1): ");
        double noise = sc.nextDouble();

        // 2️ Ask for regression type
        System.out.print("Regression type (linear/polynomial): ");
        String type = sc.next();

        int degree = 1; // default for linear regression
        if (type.equalsIgnoreCase("polynomial")) {
            System.out.print("Polynomial degree: ");
            degree = sc.nextInt();
        }

        // 3️ Generate training dataset
        Dataset train = RegressionDataGenerator.generateDataset(nSamples, nFeatures, noise);

        // 4️ Expand features if polynomial regression
        double[][] X_train = (type.equalsIgnoreCase("polynomial")) ?
                PolynomialFeatures.transform(train.X, degree) :
                train.X;

        // 5️ Fit model using Least Squares
        LeastSquares ls = new LeastSquares();
        double[] w_estimated = ls.LeastSquaresEstimator(X_train, train.Y);

        // 6️ Display estimated weights
        System.out.println("\nEstimated weights:");
        for (double w : w_estimated) System.out.print(w + " ");
        System.out.println();

        // 7️ Ask user how many test samples to generate
        System.out.print("\nNumber of test samples: ");
        int nTestSamples = sc.nextInt();

        // 8️ Generate test dataset (using the same true weights)
        Dataset test = RegressionDataGenerator.generateNewData(nTestSamples, train.trueWeights, noise);

        // 9️ Expand test features if polynomial regression
        double[][] X_test = (type.equalsIgnoreCase("polynomial")) ?
                PolynomialFeatures.transform(test.X, degree) :
                test.X;

        // 10️ Predict
        double[] Y_pred = new double[X_test.length];
        for (int i = 0; i < X_test.length; i++) {
            double y = 0;
            for (int j = 0; j < w_estimated.length; j++) {
                y += X_test[i][j] * w_estimated[j];
            }
            Y_pred[i] = y;
        }

        // 11️ Display predictions vs actual values
        System.out.println("\nPredictions on test data:");
        for (int i = 0; i < Y_pred.length; i++) {
            System.out.printf("Predicted: %.3f\tActual: %.3f\n", Y_pred[i], test.Y[i]);
        }

        // 12️ Compute Mean Squared Error (MSE)
        double mse = 0;
        for (int i = 0; i < Y_pred.length; i++) {
            double error = Y_pred[i] - test.Y[i];
            mse += error * error;
        }
        mse /= Y_pred.length;
        System.out.printf("\nMean Squared Error (MSE) on test data: %.5f\n", mse);

        // 13️ Compute R^2 (coefficient of determination)
        double meanY = 0;
        for (double y : test.Y) meanY += y;
        meanY /= test.Y.length;

        double ssTot = 0;
        double ssRes = 0;
        for (int i = 0; i < test.Y.length; i++) {
            ssTot += (test.Y[i] - meanY) * (test.Y[i] - meanY); // total variance
            ssRes += (test.Y[i] - Y_pred[i]) * (test.Y[i] - Y_pred[i]); // residual variance
        }

        double r2 = 1 - (ssRes / ssTot);
        System.out.printf("Coefficient of determination (R²) on test data: %.5f\n", r2);

        sc.close();
    }
}
