public class Centroid {

    private double x1,x2;

    public Centroid(double x1, double x2) {
        this.x1 = x1;
        this.x2 = x2;
    }

    public double getX1() { return x1; }
    public double getX2() { return x2; }


    public void moveTo(double newX1, double newX2) {
        this.x1 = newX1;
        this.x2 = newX2;
    }



}
