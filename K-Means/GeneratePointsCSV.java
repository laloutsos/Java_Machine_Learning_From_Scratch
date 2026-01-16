import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class GeneratePointsCSV {

    public static void main(String[] args) {
        String filename = "points.csv";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(filename)) {

            // Header CSV
            writer.write("x1,x2\n");

            // 1) 150 points in [0.75,1.25] x [0.75,1.25]
            generateSquare(writer, rand, 150, 0.75, 1.25, 0.75, 1.25);

            // 2) 150 points in [0,0.5] x [0,0.5]
            generateSquare(writer, rand, 150, 0, 0.5, 0, 0.5);

            // 3) 150 points in [0,0.5] x [1.5,2]
            generateSquare(writer, rand, 150, 0, 0.5, 1.5, 2);

            // 4) 150 points in [1.5,2] x [0,0.5]
            generateSquare(writer, rand, 150, 1.5, 2, 0, 0.5);

            // 5) 150 points in [1.5,2] x [1.5,2]
            generateSquare(writer, rand, 150, 1.5, 2, 1.5, 2);

            // 6) 75 points in [0.6,0.8] x [0,0.4]
            generateSquare(writer, rand, 75, 0.6, 0.8, 0, 0.4);

            // 7) 75 points in [0.6,0.8] x [1.6,2]
            generateSquare(writer, rand, 75, 0.6, 0.8, 1.6, 2);

            // 8) 75 points in [1.2,1.4] x [0,0.4]
            generateSquare(writer, rand, 75, 1.2, 1.4, 0, 0.4);

            // 9) 75 points in [1.2,1.4] x [1.6,2]
            generateSquare(writer, rand, 75, 1.2, 1.4, 1.6, 2);

            // 10) 150 points in [0,2] x [0,2] (uniform background noise)
            generateSquare(writer, rand, 150, 0, 2, 0, 2);

            System.out.println("File " + filename + " successfully created!");

        } catch (IOException e) {
            System.out.println("Error.");
            e.printStackTrace();
        }
    }

    public static void generateSquare(FileWriter writer, Random rand,
                                      int count,
                                      double xMin, double xMax,
                                      double yMin, double yMax) throws IOException {

        for (int i = 0; i < count; i++) {
            double x = xMin + (xMax - xMin) * rand.nextDouble();
            double y = yMin + (yMax - yMin) * rand.nextDouble();
            writer.write(x + "," + y + "\n");
        }
    }
}
