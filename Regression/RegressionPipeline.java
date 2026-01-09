/**
 * Simple Pipeline for Polynomial Regression on Real-World Data
 *
 * This class demonstrates a clean pipeline for solving regression problems using
 * Cross-Validation with Polynomial Regression. It is designed to be reusable for
 * any regression dataset, provided that the data is properly preprocessed and cleaned.
 *
 * Example Dataset:
 *   Real Estate Valuation Data Set (UCI Machine Learning Repository), check HousePriceEstimator.java
 *   URL: https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
 *
 * Functionality:
 *   - Takes input features and target values as parameters
 *   - Performs k-fold Cross-Validation
 *   - Fits and tests Polynomial Regression models of various degrees
 *   - Evaluates and returns the model with the best performance (e.g., lowest MSE)
 *  
 *
 * Usage:
 *   Call the provided method from your main program with your dataset.
 *   The pipeline will automatically handle model training, validation, and selection.
 * 
 */
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;



public class RegressionPipeline {


    public static Dataset loadCSV(String filePath) throws IOException {
        ArrayList<double[]> featuresList = new ArrayList<>();
        ArrayList<Double> targetList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine(); // skip header
            if (line == null) {
                throw new IOException("CSV file is empty");
            }

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue; // skip empty lines
                String[] tokens = line.split(","); // split by comma

                double[] features = new double[tokens.length - 2]; 
                // skip No (index 0) and take X1..Xn (indexes 1..n)
                for (int i = 1; i < tokens.length - 1; i++) {
                    features[i - 1] = Double.parseDouble(tokens[i]);
                }
                featuresList.add(features);

                double y = Double.parseDouble(tokens[tokens.length - 1]);
                targetList.add(y);
            }
        }

        // Convert ArrayLists to arrays
        double[][] X = new double[featuresList.size()][];
        double[] Y = new double[targetList.size()];

        for (int i = 0; i < featuresList.size(); i++) {
            X[i] = featuresList.get(i);
            Y[i] = targetList.get(i);
        }

        return new Dataset(X, Y, null); // trueWeights = null
    }

    /**
     * Splits a Dataset into training and test sets.
     * @param dataset The original dataset
     * @param trainFraction Fraction of samples to use for training (e.g., 0.7)
     * @param seed Random seed for reproducibility
     * @return An array of two Datasets: [trainDataset, testDataset]
     */
    public static Dataset[] trainTestSplit(Dataset dataset, double trainFraction) {
        int nSamples = dataset.X.length;
        int nTrain = (int) (nSamples * trainFraction);

        // Create array of indices and shuffle
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) indices[i] = i;

        Random rnd = new Random();
        for (int i = nSamples - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        // Allocate arrays for train/test
        double[][] X_train = new double[nTrain][dataset.X[0].length];
        double[] Y_train = new double[nTrain];

        double[][] X_test = new double[nSamples - nTrain][dataset.X[0].length];
        double[] Y_test = new double[nSamples - nTrain];

        // Fill training set
        for (int i = 0; i < nTrain; i++) {
            X_train[i] = dataset.X[indices[i]];
            Y_train[i] = dataset.Y[indices[i]];
        }

        // Fill test set
        for (int i = nTrain; i < nSamples; i++) {
            X_test[i - nTrain] = dataset.X[indices[i]];
            Y_test[i - nTrain] = dataset.Y[indices[i]];
        }

        Dataset trainDataset = new Dataset(X_train, Y_train, null);
        Dataset testDataset = new Dataset(X_test, Y_test, null);

        return new Dataset[]{trainDataset, testDataset};
    }

    public double[] fit(int k, int maxDegree, Dataset dt){

        // split the dataset into train and test
        Dataset[] dt_splited = trainTestSplit(dt, 0.7);

        Dataset train = dt_splited[0];
        Dataset test = dt_splited[1];

        // Perform cross Validation and find the best degree for polynomial regression 
        CVResult cvResult = CrossValidation.crossValidatePolynomial(train, maxDegree, k);

        int degree = cvResult.bestDegree;



        // Transform the features accordingly in order to perform polynomial regression 
        double[][] X_train = PolynomialFeatures.transform(train.X, degree);
        double[][] X_test = PolynomialFeatures.transform(test.X, degree);

        // Train the model by finding the best weights 
        LeastSquares ls = new LeastSquares();
        double[] w_estimated = ls.LeastSquaresEstimator(X_train, train.Y);

        System.out.println("\nEstimated weights:");
        for (double w : w_estimated) System.out.print(w + " ");
        System.out.println();

        // Make predictions - Validate your model 
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
        System.out.println("\nBest polynomial degree from CV: " + degree);
        System.out.printf("Cross-validation MSE: %.5f\n", cvResult.bestMSE);
        System.out.printf("Coefficient of determination (R²) on test data: %.5f\n", r2);

        return w_estimated;

    }

    public void predict(double[] w, Dataset dt, int degree) {

        // Transform the features using the same polynomial degree as training
        double[][] X_poly = PolynomialFeatures.transform(dt.X, degree);

        // Make predictions
        double[] Y_pred = new double[X_poly.length];
        for (int i = 0; i < X_poly.length; i++) {
            double y = 0;
            for (int j = 0; j < w.length; j++) {
                y += X_poly[i][j] * w[j];
            }
            Y_pred[i] = y;
        }

        // Display predictions
        System.out.println("\nPredictions:");
        for (int i = 0; i < Y_pred.length; i++) {
            if (dt.Y != null) {
                System.out.printf("Predicted: %.3f\tActual: %.3f\n", Y_pred[i], dt.Y[i]);
            } else {
                System.out.printf("Predicted: %.3f\n", Y_pred[i]);
            }
        }

        // Optional: compute MSE and R^2 if Y exists
        if (dt.Y != null) {
            double mse = 0;
            double meanY = 0;
            for (double y : dt.Y) meanY += y;
            meanY /= dt.Y.length;

            double ssTot = 0;
            double ssRes = 0;
            for (int i = 0; i < dt.Y.length; i++) {
                double error = Y_pred[i] - dt.Y[i];
                mse += error * error;
                ssTot += (dt.Y[i] - meanY) * (dt.Y[i] - meanY);
                ssRes += (dt.Y[i] - Y_pred[i]) * (dt.Y[i] - Y_pred[i]);
            }
            mse /= dt.Y.length;
            double r2 = 1 - (ssRes / ssTot);

            System.out.printf("\nMean Squared Error (MSE): %.5f\n", mse);
            System.out.printf("Coefficient of determination (R²): %.5f\n", r2);
        }
    }


}
