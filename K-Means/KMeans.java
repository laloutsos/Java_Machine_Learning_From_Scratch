import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class KMeans {

    public static ArrayList<Point> readPointsFromCSV(String filename) {
        ArrayList<Point> points = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            br.readLine(); // Skip header

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double x1 = Double.parseDouble(values[0]);
                double x2 = Double.parseDouble(values[1]);

                points.add(new Point(x1, x2, -1)); // -1 <=> no cluster yet
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return points;
    }

    public static ArrayList<Centroid> initializeCentroids(ArrayList<Point> points, int k) {
        ArrayList<Centroid> centroids = new ArrayList<>();
        Random rand = new Random();
        ArrayList<Integer> chosenIndices = new ArrayList<>();

        while (centroids.size() < k) {
            int index = rand.nextInt(points.size());
            if (!chosenIndices.contains(index)) {
                Point p = points.get(index);
                centroids.add(new Centroid(p.getX1(), p.getX2()));
                chosenIndices.add(index);
            }
        }

        return centroids;
    }


    public static double distance(Point p, Centroid c) {
        double dx = p.getX1() - c.getX1();
        double dy = p.getX2() - c.getX2();
        return Math.sqrt(dx*dx + dy*dy);
    }

    public static void assignPointsToClusters(
            ArrayList<Point> points,
            ArrayList<Centroid> centroids) {

        for (Point p : points) {
            double minDist = Double.MAX_VALUE;
            int bestCluster = -1;

            for (int j = 0; j < centroids.size(); j++) {
                double dist = distance(p, centroids.get(j));
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            p.setCluster(bestCluster);
        }
    }

    public static void updateCentroids(ArrayList<Point> points, ArrayList<Centroid> centroids) {
        int k = centroids.size();

        double[] sumX1 = new double[k];
        double[] sumX2 = new double[k];
        int[] count = new int[k];

        for (Point p : points) {
            int clusterId = p.getCluster();
            sumX1[clusterId] += p.getX1();
            sumX2[clusterId] += p.getX2();
            count[clusterId]++;
        }

        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                double meanX1 = sumX1[i] / count[i];
                double meanX2 = sumX2[i] / count[i];
                centroids.get(i).moveTo(meanX1, meanX2);
            }
        }
    }

    public static boolean centroidsChanged(ArrayList<Centroid> oldCentroids, ArrayList<Centroid> newCentroids, double threshold) {
        for (int i = 0; i < oldCentroids.size(); i++) {
            double dx = oldCentroids.get(i).getX1() - newCentroids.get(i).getX1();
            double dy = oldCentroids.get(i).getX2() - newCentroids.get(i).getX2();
            if (Math.sqrt(dx*dx + dy*dy) > threshold) {
                return true;
            }
        }
        return false;
    }

    public static  ArrayList<Centroid> runKMeans(ArrayList<Point> points, int k) {
        ArrayList<Centroid> centroids = initializeCentroids(points, k);
        ArrayList<Centroid> oldCentroids = new ArrayList<>();

        double threshold = 1e-9;
        int iterations = 0;

        while (true) {
            iterations++;

            oldCentroids.clear();
            for (Centroid c : centroids) {
                oldCentroids.add(new Centroid(c.getX1(), c.getX2()));
            }

            assignPointsToClusters(points, centroids);

            updateCentroids(points, centroids);

            if (!centroidsChanged(oldCentroids, centroids, threshold)) {
                System.out.println("Converged after " + iterations + " iterations.");
                break;
            }
        }

        for (int i = 0; i < centroids.size(); i++) {
            System.out.println("Centroid " + i + ": (" + centroids.get(i).getX1() + ", " + centroids.get(i).getX2() + ")");
        }

        double error = computeClusteringError(points, centroids);
        System.out.println("Total clustering error: " + error);

        return centroids;

    }

    public static double computeClusteringError(ArrayList<Point> points, ArrayList<Centroid> centroids) {
        double error = 0.0;

        for (Point p : points) {
            Centroid c = centroids.get(p.getCluster());
            double dx = p.getX1() - c.getX1();
            double dy = p.getX2() - c.getX2();
            error += Math.sqrt(dx*dx + dy*dy); // ||xi - μk||
        }

        return error;
    }

    public static void writeCentroidsToCSV(ArrayList<Centroid> centroids, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("x1,x2\n");
            for (Centroid c : centroids) {
                writer.write(c.getX1() + "," + c.getX2() + "\n");
            }
            System.out.println("Centroids saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writePointsToCSV(ArrayList<Point> points, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("x1,x2,cluster\n");
            for (Point p : points) {
                writer.write(p.getX1() + "," + p.getX2() + "," + p.getCluster() + "\n");
            }
            System.out.println("Points with cluster assignments saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args){
        String inputFile = "points.csv";
        String pointsOutputFile = "points_clustered.csv";
        String centroidsOutputFile = "centroids.csv";
        int k = -1;

        Scanner scanner = new Scanner(System.in);


        while (true) {
            System.out.println("define the number of clusters using the <define(M)> order or type exit to end program");
            String input = scanner.nextLine().trim(); // Διαγράφουμε κενά περιττά

            if (input.equalsIgnoreCase("exit")) {
                System.out.println("Exiting program.");
                break;
            }

            if (input.startsWith("define(") && input.endsWith(")")) {
                String numberStr = input.substring(7, input.length() - 1);
                try {
                    k = Integer.parseInt(numberStr);
                    System.out.println("M = " + k);
                    break;
                } catch (NumberFormatException e) {
                    System.out.println("Not a valid number! Please try again.");
                }
            } else {
                System.out.println("Invalid format! Please try again.");
            }
        }

        scanner.close();

        if (k>0){

            ArrayList<Point> points = readPointsFromCSV(inputFile);
            ArrayList<Centroid> centroids = runKMeans(points, k);
            //writePointsToCSV(points, pointsOutputFile);
            //writeCentroidsToCSV(centroids, centroidsOutputFile);
            double avgError = computeClusteringError(points, centroids) / points.size();
            System.out.println("Average distance per point: " + avgError);

        }

    }

}
