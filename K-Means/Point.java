public class Point {
    private double x1, x2;
    private int cluster;

    public Point(double x1, double x2, int cluster) {
        this.x1 = x1;
        this.x2 = x2;
        this.cluster = cluster;
    }

    public int getCluster(){
        return cluster;
    }

    public double getX1(){
        return x1;
    }

    public double getX2(){
        return x2;
    }

    public void setCluster(int cluster){
        this.cluster = cluster;
    }
}
