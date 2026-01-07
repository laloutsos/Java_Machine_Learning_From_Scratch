
public class Dataset {
    public double[][] X;      // Feature matrix
    public double[] Y;        // Target vector
    public double[] trueWeights; // True weights used to generate Y

    public Dataset(double[][] X, double[] Y, double[] trueWeights) {
        this.X = X;
        this.Y = Y;
        this.trueWeights = trueWeights;
    }
}
