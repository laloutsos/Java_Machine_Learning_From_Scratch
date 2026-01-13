import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import java.util.Set;
import java.util.*;
import java.util.Map;
import java.util.HashMap;
import java.util.function.DoubleUnaryOperator;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.io.PrintWriter;
import java.io.File;

public class Perceptron_Classifier {

    private static final Logger LOGGER = Logger.getLogger(Perceptron_Classifier.class.getName());

    private final int d ; // Number of inputs
    private final  int K ; // Number of Categories
    //private final int H1; // Number of neurons of the first hidden layer
    //private final int H2; // Number of neurons of the second hidden layer
    //private final int H3; // Number of neurons of the third hidden layer
    private final int[] H; // Here will be stored number of neurons of each hidden layer
    //An array of Activation functions that each of them  will be applied seperately to every hidden layer
    private final String[] activation_func ;
    private final int batch_size; //batch size
    private final int  max_Epochs; // number of max epochs while training
    private final double error_thresshold; // error thresshold
    private final double learning_rate; // learning rate

    // constructor of the main program
    public Perceptron_Classifier(int batch_size, int max_Epochs,
                                 double error_thresshold, double learning_rate, int d, int K, String[] activation_func, int[] H){
        this.d = d;
        this.K = K;
        this.activation_func = activation_func;
        this.H = H;
        this. batch_size = batch_size;
        this.max_Epochs = max_Epochs;
        this.error_thresshold = error_thresshold;
        this.learning_rate = learning_rate;
    }
    // stores all the csv file to a List
    public static List<String[]> readCSV(String path) {
        List<String[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                data.add(values);
            }

        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error: " + path, e);

        }

        return data;
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


    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);
        System.out.println("\n----------------------------------------------------");
        System.out.println("                   Command Format                   ");
        System.out.println("----------------------------------------------------");
        System.out.println("Use the command:");
        System.out.println();
        System.out.println("  define(batchSize, maxEpochs, errorThreshold,");
        System.out.println("         learningRate, d, K,");
        System.out.println("         neuronsLayer1, activation1,");
        System.out.println("         neuronsLayer2, activation2, ..., outputActivation)");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  define(32, 200, 0.001, 0.01, 10, 4, 8, relu, 8, tanh(u), sigmoid)");
        System.out.println("----------------------------------------------------\n");

        System.out.println("Type 'exit' to quit");
        while(true){
            String line = scanner.nextLine().trim();

            if (line.equalsIgnoreCase("exit")){ break; }
            if (line.startsWith("define(") && line.endsWith(")")){
                // we only need the values inside the parenthesis
                String params = line.substring(7, line.length() - 1);
                String[] argsArray = params.split(",");
                //if (argsArray.length != 6) {
                    //System.out.println("Error: define requires 6 parameters");
                    //continue;
               // }
                // int H1; int H2; int H3;
                int L = (argsArray.length - 6) / 2;  // number of layers
                int d; int K;  String[] activation_func = new String[L+1]; int[] H = new int[L]; int batch_size; int max_Epochs;
                double learning_rate; double error_thresshold;
                //activation_func = argsArray[2].trim();
                Set<String> valid = Set.of("relu", "tanh(u)", "sigmoid");

               // if (!valid.contains(activation_func)) {
                   // System.out.println("Choose an activation function between relu,tanh(u) or sigmoid");
                   // continue;
               // }
                try{
                    batch_size = Integer.parseInt(argsArray[0].trim());
                    max_Epochs = Integer.parseInt(argsArray[1].trim());
                    error_thresshold = Double.parseDouble(argsArray[2].trim());
                    learning_rate = Double.parseDouble(argsArray[3].trim());


                    d = Integer.parseInt(argsArray[4].trim());
                    K = Integer.parseInt(argsArray[5].trim());

                    for (int i = 0; i < L; i++) {
                        H[i] = Integer.parseInt(argsArray[6 + 2 * i].trim());
                        activation_func[i] = argsArray[7 + 2 * i].trim();
                    }
                    activation_func[activation_func.length-1] = argsArray[argsArray.length-1];


                }  catch (NumberFormatException e) {
                    System.out.println("Error: The first 5 parameters must be integers");
                    continue;
                }
                // we create a new object that contains all the parameters we need
                Perceptron_Classifier classifier = new Perceptron_Classifier(batch_size, max_Epochs, error_thresshold, learning_rate,
                        d, K, activation_func, H);

                System.out.println("\n====================================================");
                System.out.println("            Neural Network Configuration            ");
                System.out.println("====================================================");
                System.out.println("Inputs (d):               " + classifier.d);
                System.out.println("Output classes (K):       " + classifier.K);
                System.out.println("Batch size:               " + classifier.batch_size);
                System.out.println("Max Epochs:               " + classifier.max_Epochs);
                System.out.println("Learning rate:            " + classifier.learning_rate);
                System.out.println("Error threshold:          " + classifier.error_thresshold);
                System.out.println();
                System.out.println("Hidden Layers:");
                for (int i = 0; i < L; i++) {
                    System.out.println("  Layer " + (i + 1) + " -> Neurons: " + H[i] +
                            " | Activation: " + activation_func[i]);
                }
                System.out.println("====================================================\n");



                DoubleUnaryOperator[] activation = new DoubleUnaryOperator[activation_func.length];
                DoubleUnaryOperator[] activationDerivative = new DoubleUnaryOperator[activation_func.length];
                for(int i=0; i<activation_func.length; i++) {
                    switch (activation_func[i].toLowerCase()) {
                        case "relu":
                            activation[i] = Neuron.RELU;
                            activationDerivative[i] = Neuron.RELU_DERIVATIVE;
                            break;
                        case "tanh(u)":
                            activation[i] = Neuron.TANH;
                            activationDerivative[i] = Neuron.TANH_DERIVATIVE;
                            break;
                        default:
                            activation[i] = Neuron.SIGMOID;
                            activationDerivative[i] = Neuron.SIGMOID_DERIVATIVE;
                    }
                }

                MultiLayerPerceptron mlp = new MultiLayerPerceptron(d, H, K, activation, activationDerivative);


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

                int n = 5;
                for (int i = 0; i < n && i < X_train.size(); i++) {
                    System.out.print("Input: ");
                    for (double val : X_train.get(i)) {
                        System.out.print(val + " ");
                    }

                    System.out.print(" | One-hot output: ");
                    for (double val : Y_train.get(i)) {
                        System.out.print((int)val + " ");
                    }

                    System.out.println();
                }

                System.out.println("Start training? Type 'yes' to continue, anything else to quit:");
                String start = scanner.nextLine().trim();

                if (!start.equalsIgnoreCase("yes")) {
                    System.out.println("Training cancelled. Exiting program.");
                    break;
                }

                mlp.train(X_train, Y_train, max_Epochs, batch_size, learning_rate, error_thresshold);
                List<String> results = new ArrayList<>();
                results = mlp.test(X_test,Y_test,mlp);
                System.out.println("Save Predicted Results? Type 'yes' to continue, anything else to quit:");
                String end = scanner.nextLine().trim();

                if (!end.equalsIgnoreCase("yes")) {
                    System.out.println("Saving results cancelled. Exiting program.");
                    break;
                }

                try (PrintWriter writer = new PrintWriter("results.csv")) {

                    // --- Write classifier metadata ---
                    writer.println("=== Classifier Parameters ===");
                    writer.println("d,K,batch_size,maxEpochs,learningRate,errorThreshold,H_layers,activations");
                    writer.print(classifier.d + "," + classifier.K + "," + classifier.batch_size + ","
                            + classifier.max_Epochs + "," + classifier.learning_rate + ","
                            + classifier.error_thresshold + ",");

                    // Layers
                    for (int i = 0; i < H.length; i++) {
                        writer.print(H[i]);
                        if (i < H.length - 1) writer.print("-");
                    }
                    writer.print(",");

                    // Activation functions
                    for (int i = 0; i < activation_func.length; i++) {
                        writer.print(activation_func[i]);
                        if (i < activation_func.length - 1) writer.print("-");
                    }
                    writer.println();

                    writer.println("=== Classifier Results ===");

                    writer.println("x1,x2,predicted,actual");

                    for (String linee : results) {

                        // Extract input array: [a,b]
                        int a = linee.indexOf("[");
                        int b = linee.indexOf("]") + 1;

                        String inputString = linee.substring(a, b); // "[1.2, 3.4]"

                        // Clean the brackets so we can split
                        String noBrackets = inputString.replace("[", "").replace("]", "");
                        String[] parts = noBrackets.split(",");

                        String x1 = parts[0].trim();
                        String x2 = parts[1].trim();

                        // Extract predicted class
                        String predicted = linee.split("\\|")[1]
                                .replace("Predicted:", "")
                                .trim();

                        // Extract actual class
                        String actual = linee.split("\\|")[2]
                                .replace("Actual:", "")
                                .trim();

                        writer.println(x1 + "," + x2 + "," + predicted + "," + actual);
                    }

                    System.out.println("CSV saved to results.csv");

                } catch (IOException e) {
                    System.out.println("Error writing CSV file: " + e.getMessage());
                }

                break;
            }
            else {
                System.out.println("Unknown command: " + line);
            }
        }

    }

}
