import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        Dataset train;
        Dataset test;
        double[][] X_train;
        double[][] X_test;
        int degree = 1;

        // 1️ Ask if user wants polynomial regression
        System.out.print("Do you want polynomial regression? (yes/no): ");
        String polyChoice = sc.next();

        boolean usePoly = polyChoice.equalsIgnoreCase("yes");

        // 2️ Ask for cross-validation
        boolean useCV = false;
        int cvFolds = 5; // default
        int maxDegree = 3; // default max degree for CV

        if (usePoly) {
            System.out.print("Do you want cross-validation to choose polynomial degree? (yes/no): ");
            String cvChoice = sc.next();
            useCV = cvChoice.equalsIgnoreCase("yes");

            if (useCV) {
                System.out.print("Maximum polynomial degree to try: ");
                maxDegree = sc.nextInt();

                System.out.print("Number of folds for cross-validation: ");
                cvFolds = sc.nextInt();
            }
        }

        if (usePoly) {
            // 3️ Ask for non-linear dataset type
            System.out.print("Choose dataset type (sin / cubic / expcos): ");
            String datasetType = sc.next();

            System.out.print("Number of training samples: ");
            int nSamples = sc.nextInt();

            System.out.print("Noise standard deviation (e.g., 0.1): ");
            double noise = sc.nextDouble();

            System.out.print("Polynomial degree (if no CV, e.g., 2): ");
            degree = sc.nextInt();

            // Generate dataset
            if (datasetType.equalsIgnoreCase("sin")) {
                train = RegressionDataGenerator.generateSinDataset(nSamples, noise);
                test = RegressionDataGenerator.generateSinDataset(nSamples, noise);
            } else if (datasetType.equalsIgnoreCase("cubic")) {
                train = RegressionDataGenerator.generatePolynomialDataset(nSamples, noise, degree);
                test = RegressionDataGenerator.generatePolynomialDataset(nSamples, noise, degree);
            } else {
                train = RegressionDataGenerator.generateExpCosDataset(nSamples, noise);
                test = RegressionDataGenerator.generateExpCosDataset(nSamples, noise);
            }

            // Cross-validation to pick degree
            if (useCV) {
                CVResult cvResult = CrossValidation.crossValidatePolynomial(train, maxDegree, cvFolds);
                degree = cvResult.bestDegree;
                System.out.println("\nBest polynomial degree from CV: " + degree);
                System.out.printf("Cross-validation MSE: %.5f\n", cvResult.bestMSE);
            }

            // Transform features
            X_train = PolynomialFeatures.transform(train.X, degree);
            X_test = PolynomialFeatures.transform(test.X, degree);

        } else {
            // ===== LINEAR REGRESSION =====
            System.out.print("Number of training samples: ");
            int nSamples = sc.nextInt();

            System.out.print("Number of features: ");
            int nFeatures = sc.nextInt();

            System.out.print("Noise standard deviation (e.g., 0.1): ");
            double noise = sc.nextDouble();

            train = RegressionDataGenerator.generateDataset(nSamples, nFeatures, noise);
            test = RegressionDataGenerator.generateNewData(nSamples, train.trueWeights, noise);

            X_train = train.X;
            X_test = test.X;
        }

        // ===== Fit model using Least Squares =====
        LeastSquares ls = new LeastSquares();
        double[] w_estimated = ls.LeastSquaresEstimator(X_train, train.Y);

        // Display estimated weights
        System.out.println("\nEstimated weights:");
        for (double w : w_estimated) System.out.print(w + " ");
        System.out.println();

        // Predict on test set
        double[] Y_pred = new double[X_test.length];
        for (int i = 0; i < X_test.length; i++) {
            double y = 0;
            for (int j = 0; j < w_estimated.length; j++) {
                y += X_test[i][j] * w_estimated[j];
            }
            Y_pred[i] = y;
        }

        // Display predictions vs actual
        System.out.println("\nPredictions on test data:");
        for (int i = 0; i < Y_pred.length; i++) {
            System.out.printf("Predicted: %.3f\tActual: %.3f\n", Y_pred[i], test.Y[i]);
        }

        // Compute MSE
        double mse = 0;
        for (int i = 0; i < Y_pred.length; i++) {
            double error = Y_pred[i] - test.Y[i];
            mse += error * error;
        }
        mse /= Y_pred.length;
        System.out.printf("\nMean Squared Error (MSE): %.5f\n", mse);

        // Compute R^2
        double meanY = 0;
        for (double y : test.Y) meanY += y;
        meanY /= test.Y.length;

        double ssTot = 0;
        double ssRes = 0;
        for (int i = 0; i < test.Y.length; i++) {
            ssTot += (test.Y[i] - meanY) * (test.Y[i] - meanY);
            ssRes += (test.Y[i] - Y_pred[i]) * (test.Y[i] - Y_pred[i]);
        }

        double r2 = 1 - (ssRes / ssTot);
        System.out.printf("Coefficient of determination (R²) on test data: %.5f\n", r2);

        sc.close();
    }
}
