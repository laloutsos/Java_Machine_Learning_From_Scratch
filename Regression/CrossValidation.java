import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CrossValidation {

    // Split dataset into k folds (indices)
    public static List<int[]> split_data_set_into_k_folds(Dataset dt, int k) {
        double[][] X = dt.X;
        int foldSize = X.length / k;
        List<int[]> folds = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            int start = i * foldSize;
            int end = (i == k - 1) ? X.length : start + foldSize;
            int[] foldIndices = new int[end - start];
            for (int j = start; j < end; j++) foldIndices[j - start] = j;
            folds.add(foldIndices);
        }

        return folds;
    }

    // Cross-validation for polynomial regression
    public static CVResult crossValidatePolynomial(Dataset train, int maxDegree, int k) {

        double bestMSE = Double.MAX_VALUE;
        int bestDegree = 1;

        List<int[]> folds = split_data_set_into_k_folds(train, k);

        for (int degree = 1; degree <= maxDegree; degree++) {
            double mseSum = 0;

            for (int f = 0; f < k; f++) {
                // Prepare train and validation sets for this fold
                List<double[]> X_train_fold = new ArrayList<>();
                List<Double> Y_train_fold = new ArrayList<>();
                List<double[]> X_val_fold = new ArrayList<>();
                List<Double> Y_val_fold = new ArrayList<>();

                for (int i = 0; i < train.X.length; i++) {
                    final int idx = i;
                    if (Arrays.stream(folds.get(f)).anyMatch(x -> x == idx)) {
                        X_val_fold.add(train.X[i]);
                        Y_val_fold.add(train.Y[i]);
                    } else {
                        X_train_fold.add(train.X[i]);
                        Y_train_fold.add(train.Y[i]);
                    }
                }

                // Convert lists to arrays
                double[][] X_train_array = X_train_fold.toArray(new double[0][]);
                double[][] X_val_array = X_val_fold.toArray(new double[0][]);
                double[] Y_train_array = Y_train_fold.stream().mapToDouble(d -> d).toArray();
                double[] Y_val_array = Y_val_fold.stream().mapToDouble(d -> d).toArray();

                // Transform features to polynomial degree
                X_train_array = PolynomialFeatures.transform(X_train_array, degree);
                X_val_array = PolynomialFeatures.transform(X_val_array, degree);

                // Train model
                LeastSquares ls = new LeastSquares();
                double[] w = ls.LeastSquaresEstimator(X_train_array, Y_train_array);

                // Predict on validation set
                double mse = 0;
                for (int i = 0; i < X_val_array.length; i++) {
                    double yPred = 0;
                    for (int j = 0; j < w.length; j++) {
                        yPred += X_val_array[i][j] * w[j];
                    }
                    double error = yPred - Y_val_array[i];
                    mse += error * error;
                }
                mse /= X_val_array.length;
                mseSum += mse;
            }

            double avgMSE = mseSum / k;
            if (avgMSE < bestMSE) {
                bestMSE = avgMSE;
                bestDegree = degree;
            }
        }

        return new CVResult(bestDegree, bestMSE);
    }
}
