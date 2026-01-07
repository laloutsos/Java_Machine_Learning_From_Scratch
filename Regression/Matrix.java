public class Matrix {

    // Returns the transpose of a table A
    public static double[][] transpose(double[][] A){
        int rows = A[0].length;
        int columns = A.length;

        double[][] T = new double[rows][columns];
        for(int i = 0; i<columns; i++){
            for(int j=0; j<rows; j++){
                T[j][i] = A[i][j];
            }
        }

        return T;
    }
    // Multiplies 2 matrixes
    public static double[][] multiply(double[][] A, double[][] B){
        // int rowsA = A[0].length;
        int columnsA = A.length;

        int rowsB = B[0].length;
        int columnsB = B.length;

        double[][] C = new double[columnsA][rowsB];

        for (int i=0; i<columnsA; i++){
            for (int j=0; j<columnsA; j++){
                for(int k = 0; k<columnsB; k++){
                    C[i][j] += A[i][k] * B[k][j];

                }
            }
        }

        return C;
    }

    public static double[] multiply(double[][] A, double[] x) {
        double[] y = new double[A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < x.length; j++)
                y[i] += A[i][j] * x[j];
        return y;
    }

    // Gauss-Jordan inverse
    public static double[][] inverse(double[][] A) {

        int n = A.length;
        double[][] aug = new double[n][2*n];

        // build [A | I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                aug[i][j] = A[i][j];
            aug[i][i+n] = 1;
        }

        // eliminate
        for (int i = 0; i < n; i++) {

            double pivot = aug[i][i];
            if (pivot == 0)
                throw new RuntimeException("Matrix not invertible");

            for (int j = 0; j < 2*n; j++)
                aug[i][j] /= pivot;

            for (int k = 0; k < n; k++) {
                if (k == i) continue;
                double factor = aug[k][i];
                for (int j = 0; j < 2*n; j++)
                    aug[k][j] -= factor * aug[i][j];
            }
        }

        // extract right half
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                inv[i][j] = aug[i][j+n];

        return inv;
    }
}
