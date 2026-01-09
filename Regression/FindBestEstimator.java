import java.io.IOException;

public class FindBestEstimator {
    public static void main(String[] args){
        Dataset dt = null;
        // load the csv file 
        try {
            dt = RegressionPipeline.loadCSV("real_estate.csv");

            System.out.println("Loaded " + dt.X.length + " rows.");

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error reading CSV file!");
        }
        //Feature Engineering 
        Dataset dt_f = HousePriceEstimator.featureEngineering(dt);
        System.out.println("After feature engineering: " + dt_f.X.length + " rows, " + dt_f.X[0].length + " columns");




        double[] best_weights = null;
        RegressionPipeline best_regression = null;
        double best_mse = Double.POSITIVE_INFINITY; 
        int best_k = 0;


        for(int k=2; k<30; k++){

            RegressionPipeline pipeline = new RegressionPipeline();
            double[] w_estimated = pipeline.fit(k, 15, dt_f);

            double test_mse = pipeline.get_test_mse();
            if(test_mse<best_mse){
                best_mse = test_mse;
                best_weights = w_estimated;
                best_regression = pipeline;
                best_k = k;
            }

        }

        int bestDegree = best_regression.getDegree();

        System.out.println("Best fold: " + best_k);




        best_regression.predict(best_weights, dt_f, bestDegree );

        

    }
}
