public class PolynomialFeatures {

    /**
     * Generates polynomial features up to the given degree.
     * @param X Original feature matrix (nSamples x nFeatures)
     * @param degree Maximum degree of polynomial
     * @return Expanded feature matrix (nSamples x nNewFeatures)
     */
    public static double[][] transform(double[][] X, int degree) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        // Calculate total number of features after polynomial expansion
        int nNewFeatures = nFeatures * degree; // simple approach: x, x^2, ..., x^degree for each feature
        double[][] X_poly = new double[nSamples][nNewFeatures];

        for (int i = 0; i < nSamples; i++) {
            int col = 0;
            for (int j = 0; j < nFeatures; j++) {
                double val = X[i][j];
                for (int d = 1; d <= degree; d++) {
                    X_poly[i][col] = Math.pow(val, d);
                    col++;
                }
            }
        }
        return X_poly;
    }
}
