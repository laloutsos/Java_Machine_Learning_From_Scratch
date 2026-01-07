public class LeastSquares {

    public double[] LeastSquaresEstimator(double[][] X, double[] Y){
        double[][] Xt = Matrix.transpose(X);          // X^T
        double[][] XtX = Matrix.multiply(Xt, X);     // X^T * X
        double[][] XtX_inv = Matrix.inverse(XtX);    // (X^T X)^-1
        double[] XtY = Matrix.multiply(Xt, Y);       // X^T * Y
        double[] w = Matrix.multiply(XtX_inv, XtY);  // w = (X^T X)^-1 X^T Y


        return w;
    }


}
