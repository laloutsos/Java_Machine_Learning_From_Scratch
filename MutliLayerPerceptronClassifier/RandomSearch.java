import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.io.PrintWriter;
import java.util.function.DoubleUnaryOperator;
import java.util.logging.Level;

public class RandomSearch {


    public static List<String[]> readCSV(String path) {
        List<String[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                data.add(values);
            }

        } catch (IOException e) {
            System.err.println("Error processing path: " + path);
            e.printStackTrace();


        }

        return data;
    }

    private static void saveBestResultsToCSV(int[] H, String[] activations,
                                             int batch, int epochs, double lr,
                                             double threshold, double accuracy,
                                             List<String> results) {

        try (PrintWriter writer = new PrintWriter("best_results6.csv")) {

            writer.println("=== BEST CLASSIFIER PARAMETERS ===");
            writer.println("Accuracy," + accuracy);
            writer.println("Layers," + Arrays.toString(H));
            writer.println("Activations," + Arrays.toString(activations));
            writer.println("Batch," + batch);
            writer.println("Epochs," + epochs);
            writer.println("LearningRate," + lr);
            writer.println("ErrorThreshold," + threshold);
            writer.println();

            writer.println("x1,x2,predicted,actual");

            for (String line : results) {
                int a = line.indexOf("[");
                int b = line.indexOf("]") + 1;

                String[] xy = line.substring(a, b)
                        .replace("[", "")
                        .replace("]", "")
                        .split(",");

                String predicted = line.split("\\|")[1]
                        .replace("Predicted:", "").trim();

                String actual = line.split("\\|")[2]
                        .replace("Actual:", "").trim();

                writer.println(xy[0].trim() + "," +
                        xy[1].trim() + "," +
                        predicted + "," +
                        actual);
            }

        } catch (Exception e) {
            System.out.println("Error saving best_results.csv: " + e.getMessage());
        }
    }

    // encodes one label
    public static double[] oneHotEncode(String label, Map<String, Integer> labelMap) {
        int K = labelMap.size();
        double[] output = new double[K];

        int index = labelMap.get(label);
        output[index] = 1.0;

        return output;
    }

    // encodes all labels
    public static void oneHotEncodeForAll(List<String[]> list,
                                          List<double[]> X_train,
                                          List<double[]> Y_train) {

        Map<String, Integer> labelMap = new HashMap<>();
        labelMap.put("C1", 0);
        labelMap.put("C2", 1);
        labelMap.put("C3", 2);
        labelMap.put("C4", 3);

        for (String[] row : list) {
            // Features
            double[] input = new double[row.length - 1];
            for (int i = 0; i < row.length - 1; i++) {
                input[i] = Double.parseDouble(row[i]);
            }

            double[] output = oneHotEncode(row[row.length - 1], labelMap);

            X_train.add(input);
            Y_train.add(output);
        }
    }

    private static void appendModelInfo(int exp, int[] H, String[] activations,
                                        int batch, int epochs, double lr,
                                        double threshold, double accuracy) {

        try (PrintWriter writer = new PrintWriter(new java.io.FileWriter("all_models6.csv", true))) {

            if (new java.io.File("all_models6.csv").length() == 0) {
                writer.println("Experiment,Accuracy,Layers,Activations,Batch,Epochs,LearningRate,ErrorThreshold");
            }

            writer.println(
                    exp + "," +
                            accuracy + "," +
                            Arrays.toString(H) + "," +
                            Arrays.toString(activations) + "," +
                            batch + "," +
                            epochs + "," +
                            lr + "," +
                            threshold
            );

        } catch (Exception e) {
            System.out.println("Error writing to all_models2.csv: " + e.getMessage());
        }
    }



    public static void main(String[] args){
        int d = 2;
        int K = 4;
        int experiments = 200;
        Random rand = new Random();

        double bestAccuracy = -1.0;
        List<String> bestResults = null;

        int[] bestH = null;
        String[] bestActivations = null;
        int bestBatch = 0;
        int bestEpochs = 0;
        double bestLR = 0.0;
        double bestThreshold = 0.0;

        List<String[]> train_set = readCSV("train.csv");
        List<String[]> test_set = readCSV("test.csv");
        List<double[]> X_train = new ArrayList<>();
        List<double[]> Y_train = new ArrayList<>();
        List<double[]> X_test = new ArrayList<>();
        List<double[]> Y_test = new ArrayList<>();

        train_set.remove(0);
        test_set.remove(0);

        oneHotEncodeForAll(train_set, X_train, Y_train);
        oneHotEncodeForAll(test_set, X_test, Y_test);

        for (int exp = 1; exp <= experiments; exp++) {

            System.out.println("\n===============================");
            System.out.println("   RANDOM SEARCH EXPERIMENT " + exp);
            System.out.println("===============================\n");

            // ----- Random hyperparameters -----
            int batch_size = new int[]{8, 16, 32, 64, 128,256,1000,2000,3000,4000}[rand.nextInt(8)];
            int max_Epochs = 4000;
            //double learning_rate = Math.pow(10, -(2 + rand.nextDouble() * 2)); // 1e-2 to 1e-4
            double learning_rate = 0.001; // 1e-2 to 1e-4

            double error_threshold = 0.000001;

            int numHiddenLayers = 3;
            int[] H = new int[numHiddenLayers];
            String[] activation_func = new String[numHiddenLayers + 1];

            String[] choices = {"tanh", "sigmoid","relu"};


            for (int i = 0; i < numHiddenLayers; i++) {
                H[i] = 4 + rand.nextInt(40);

                activation_func[i] = choices[rand.nextInt(choices.length)];
            }

            activation_func[numHiddenLayers] = choices[rand.nextInt(choices.length)]; // output layer


            // Build activation operators
            DoubleUnaryOperator[] activation = new DoubleUnaryOperator[activation_func.length];
            DoubleUnaryOperator[] activationDerivative = new DoubleUnaryOperator[activation_func.length];

            for (int i = 0; i < activation_func.length; i++) {
                switch (activation_func[i].toLowerCase()) {
                    case "relu":
                        activation[i] = Neuron.RELU;
                        activationDerivative[i] = Neuron.RELU_DERIVATIVE;
                        break;
                    case "tanh":
                        activation[i] = Neuron.TANH;
                        activationDerivative[i] = Neuron.TANH_DERIVATIVE;
                        break;
                    case "sigmoid":
                    default:
                        activation[i] = Neuron.SIGMOID;
                        activationDerivative[i] = Neuron.SIGMOID_DERIVATIVE;
                        break;
                }
            }

            // ---- Build the MLP ----
            MultiLayerPerceptron mlp =
                    new MultiLayerPerceptron(d, H, K, activation, activationDerivative);

            // ---- Train ----
            mlp.train(X_train, Y_train, max_Epochs, batch_size, learning_rate, error_threshold);

            // ---- Evaluate ----
            List<String> results = mlp.test(X_test, Y_test, mlp);
            double accuracy = mlp.getAccuracy();

            System.out.println("ACCURACY = " + accuracy);



            appendModelInfo(exp, H, activation_func, batch_size, max_Epochs, learning_rate, error_threshold, accuracy);


                // ---- Keep best only ----
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestResults = results;
                bestH = Arrays.copyOf(H, H.length);
                bestActivations = Arrays.copyOf(activation_func, activation_func.length);
                bestBatch = batch_size;
                bestEpochs = max_Epochs;
                bestLR = learning_rate;
                bestThreshold = error_threshold;

                System.out.println(">>> New BEST model found!");
            }
        }

// ---- After all experiments, save best model only ----

        saveBestResultsToCSV(bestH, bestActivations, bestBatch, bestEpochs,
                bestLR, bestThreshold, bestAccuracy, bestResults);

        System.out.println("\n====================================");
        System.out.println(" BEST MODEL SAVED TO best_results.csv");
        System.out.println(" Accuracy = " + bestAccuracy);
        System.out.println("====================================\n");

    }
}
