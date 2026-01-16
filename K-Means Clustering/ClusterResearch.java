import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class ClusterResearch {

    public static void main(String[] args) {

        int[] M_values = {3, 5, 7, 9, 11, 13};
        String inputFile = "points.csv";

        for (int M : M_values) {
            double bestError = Double.MAX_VALUE;
            ArrayList<Centroid> bestCentroids = null;
            ArrayList<Point> bestPoints = null;

            System.out.println("Running k-means for M = " + M);

            for (int run = 1; run <= 50; run++) {
                ArrayList<Point> points = KMeans.readPointsFromCSV(inputFile);
                ArrayList<Centroid> centroids = KMeans.runKMeans(points, M);

                double error = KMeans.computeClusteringError(points, centroids);
                System.out.println("Run " + run + ": error = " + error);

                if (error < bestError) {
                    bestError = error;

                    bestCentroids = new ArrayList<>();
                    for (Centroid c : centroids) {
                        bestCentroids.add(new Centroid(c.getX1(), c.getX2()));
                    }

                    bestPoints = new ArrayList<>();
                    for (Point p : points) {
                        bestPoints.add(new Point(p.getX1(), p.getX2(), p.getCluster()));
                    }
                }
            }
            String centroidsFile = "best_centroids_M" + M + ".csv";
            KMeans.writeCentroidsToCSV(bestCentroids, centroidsFile);
            try (FileWriter writer = new FileWriter("best_error_M" + M + ".txt")) {
                writer.write("Best error for M=" + M + ": " + bestError);
            } catch (IOException e) {
                e.printStackTrace();
            }

            System.out.println("Best error for M=" + M + ": " + bestError);
            System.out.println("--------------------------------------------------");
        }
    }
}
