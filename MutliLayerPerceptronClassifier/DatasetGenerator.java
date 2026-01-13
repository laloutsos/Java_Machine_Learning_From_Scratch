import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

public class DatasetGenerator {

    private static final int TOTAL = 10000;
    private static final int TRAIN = 10000;
    private static final int TEST = 10000;



    public static void main(String[] args) {
        String trainFile = "trainnnn.csv";
        String testFile = "testnnn.csv";

        Random rnd = new Random();

        try (BufferedWriter trainWriter = new BufferedWriter(new FileWriter(trainFile));
             BufferedWriter testWriter = new BufferedWriter(new FileWriter(testFile))) {

            // CSV header
            trainWriter.write("x1,x2,label\n");
            testWriter.write("x1,x2,label\n");

            // counters
            Map<String, Integer> trainCounts = new HashMap<>();
            Map<String, Integer> testCounts = new HashMap<>();
            initializeCounts(trainCounts);
            initializeCounts(testCounts);

            for (int i = 0; i < TOTAL; i++) {
                double x1 = rnd.nextDouble(2.0);
                double x2 = rnd.nextDouble(2.0);
                String label = classify(x1, x2);

                String line = String.format("%.6f,%.6f,%s\n", x1, x2, label);

                //if (i < TRAIN) {
                 //   trainWriter.write(line);
                 //   trainCounts.put(label, trainCounts.get(label) + 1);
               // } else {
                    testWriter.write(line);
                    testCounts.put(label, testCounts.get(label) + 1);
               // }
            }

            System.out.println("Generated " + TRAIN + " training samples -> " + trainFile);
            System.out.println("Generated " + TEST + " test samples     -> " + testFile);
            System.out.println();

            //System.out.println("Class distribution (training set):");
            //printCounts(trainCounts);

            System.out.println("Class distribution (test set):");
            printCounts(testCounts);

        } catch (IOException e) {
            System.err.println("Error generating files: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void initializeCounts(Map<String, Integer> m) {
        m.put("C1", 0);
        m.put("C2", 0);
        m.put("C3", 0);
        m.put("C4", 0);
    }

    private static boolean inCircle(double x1, double x2, double cx, double cy, double r2) {
        double dx = x1 - cx;
        double dy = x2 - cy;
        return dx*dx + dy*dy < r2;
    }

    public static String classify(double x1, double x2) {
        double r2 = 0.2;

        // circle (0.5, 0.5)
        if (inCircle(x1, x2, 0.5, 0.5, r2)) {
            if (x1 > 0.5 && x2 > 0.5) return "C1";
            if (x1 < 0.5 && x2 > 0.5) return "C2";
            if (x1 > 0.5 && x2 < 0.5) return "C2";
            if (x1 < 0.5 && x2 < 0.5) return "C1";
        }

        // circle (1.5, 0.5)
        if (inCircle(x1, x2, 1.5, 0.5, r2)) {
            if (x1 > 1.5 && x2 > 0.5) return "C1";
            if (x1 < 1.5 && x2 > 0.5) return "C2";
            if (x1 > 1.5 && x2 < 0.5) return "C2";
            if (x1 < 1.5 && x2 < 0.5) return "C1";
        }

        // circle (0.5, 1.5)
        if (inCircle(x1, x2, 0.5, 1.5, r2)) {
            if (x1 > 0.5 && x2 > 1.5) return "C1";
            if (x1 < 0.5 && x2 > 1.5) return "C2";
            if (x1 > 0.5 && x2 < 1.5) return "C2";
            if (x1 < 0.5 && x2 < 1.5) return "C1";
        }

        // circle (1.5, 1.5)
        if (inCircle(x1, x2, 1.5, 1.5, r2)) {
            if (x1 > 1.5 && x2 > 1.5) return "C1";
            if (x1 < 1.5 && x2 > 1.5) return "C2";
            if (x1 > 1.5 && x2 < 1.5) return "C2";
            if (x1 < 1.5 && x2 < 1.5) return "C1";
        }

        // rules 17–18
        double prod = (x1 - 1.0) * (x2 - 1.0);

        if (prod > 0) return "C3";  // rule 17
        return "C4";               // rule 18
    }

    private static void printCounts(Map<String, Integer> counts) {
        System.out.println("C1: " + counts.get("C1"));
        System.out.println("C2: " + counts.get("C2"));
        System.out.println("C3: " + counts.get("C3"));
        System.out.println("C4: " + counts.get("C4"));
        System.out.println();
    }
}
