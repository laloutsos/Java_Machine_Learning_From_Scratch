public class CVResult {
    public int bestDegree;
    public double bestMSE;
    public CVResult(int degree, double mse) {
        this.bestDegree = degree;
        this.bestMSE = mse;
    }
}
