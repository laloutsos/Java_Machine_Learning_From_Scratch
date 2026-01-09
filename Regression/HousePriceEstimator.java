import java.io.IOException;
import java.util.Scanner;

public class HousePriceEstimator {

    public static Dataset featureEngineering(Dataset dt) {
        int nRows = dt.X.length;
        int nCols = dt.X[0].length;

        // Create new array for transformed features
        double[][] X_new = new double[nRows][nCols];

        //  Standardize all features (mean 0, std 1)
        double[] means = new double[nCols];
        double[] stds = new double[nCols];

        // Compute mean
        for (int j = 0; j < nCols; j++) {
            double sum = 0;
            for (int i = 0; i < nRows; i++) sum += dt.X[i][j];
            means[j] = sum / nRows;
        }

        // Compute std
        for (int j = 0; j < nCols; j++) {
            double sumSq = 0;
            for (int i = 0; i < nRows; i++) sumSq += Math.pow(dt.X[i][j] - means[j], 2);
            stds[j] = Math.sqrt(sumSq / nRows);
            if (stds[j] == 0) stds[j] = 1; // avoid division by zero
        }

        // Apply standardization
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                X_new[i][j] = (dt.X[i][j] - means[j]) / stds[j];
            }
        }

        // 2 Center the transaction date (X1)
        for (int i = 0; i < nRows; i++) {
            X_new[i][0] -= 2012.5;  // subtract mean year for centering
        }


        //  Return new Dataset with transformed features
        return new Dataset(X_new, dt.Y, dt.trueWeights);
    }


    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // 1️ Ask user for training CSV path
        System.out.print("Enter path to training CSV file: ");
        String csvPath = sc.nextLine();
        Dataset dt = null;
        // load the csv file 
        try {
            dt = RegressionPipeline.loadCSV(csvPath);

            System.out.println("Loaded " + dt.X.length + " rows.");

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error reading CSV file!");
        }
        //Feature Engineering 
        Dataset dt_f = featureEngineering(dt);


        // 2️ Ask for k (cross-validation folds)
        System.out.print("Enter number of folds for Cross-Validation (k): ");
        int k = sc.nextInt();

        // 3️ Ask for maximum polynomial degree
        System.out.print("Enter maximum polynomial degree: ");
        int maxDegree = sc.nextInt();

        // 4️ Run the pipeline
        RegressionPipeline pipeline = new RegressionPipeline();

        double[] w_estimated = pipeline.fit(k, maxDegree, dt_f);

        // 5️ Optionally, predict on another CSV
        sc.nextLine(); // consume leftover newline
        System.out.print("\nEnter path to CSV file for prediction (or leave empty to skip): ");
        String predictPath = sc.nextLine();

        if (!predictPath.isEmpty()) {
            System.out.print("Enter polynomial degree to use for prediction: ");
            int degree = sc.nextInt();
            pipeline.predict(w_estimated, dt_f, degree);
        }

        sc.close();
        System.out.println("\n=== Pipeline Finished ===");
    }
}
