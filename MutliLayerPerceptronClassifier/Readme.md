## 📂 Project Structure & Class Overview

| Class | Purpose / Description |
|-------|----------------------|
| **Neuron** | Represents a single artificial neuron. Stores weights, bias, activation function, and supports forward and backward propagation with gradient-based learning. |
| **Layer** | Represents a neural network layer composed of multiple neurons. Handles forward propagation, backward propagation of gradients, parameter updates, and gradient accumulation across neurons. |
| **MultiLayerPerceptron** | Implements a feedforward neural network composed of multiple layers. Supports forward propagation, backpropagation, mini-batch gradient descent training, and performance evaluation on test data. |
| **Perceptron_Classifier** | Acts as the main application controller for configuring, training, testing, and evaluating a multilayer perceptron classifier. Handles user input, data loading, preprocessing, model initialization, training execution, evaluation, and result persistence. |
| **RandomSearch** | Implements random hyperparameter search for a multilayer perceptron classifier. Repeatedly trains and evaluates models with randomly selected configurations, tracks performance, and stores both all experiments and the best-performing model results to CSV files. |
| **DatasetGenerator** | Generates a synthetic two-dimensional classification dataset by sampling random points and assigning class labels based on geometric rules. Outputs labeled data to CSV files for training and testing purposes. |
| **ModelAnalysis (Python Script)** | Performs post-training analysis of multiple neural network experiments by loading model statistics from CSV files, extracting architectural and hyperparameter features, and generating visualizations to study their impact on classification accuracy. |


** Find full Project Analysis at: **
[PDF](MLPReport.pdf)

---

### Run Instructions

1. Compile:

```
javac *.java
```

2. Run the main program:

```
java Perceptron_Classifier
```

---

### `define` Format

```
define(
  number of batches,
  max epochs,
  error threshold,
  learning rate,
  input size,
  number of categories,
  number of neurons, activation function,
  number of neurons, activation function,
  ...
  output activation function
)
```

---

### Example `define` Configurations

```java
define(8, 2000, 0.0000001, 0.01, 2, 4,
       12, tanh(u),
       12, tanh(u),
       12, tanh(u),
       sigmoid);
```

```java
define(1, 2000, 0.0000001, 0.01, 2, 4,
       12, tanh(u),
       12, tanh(u),
       12, tanh(u),
       tanh(u));
```

```java
define(16, 2000, 0.0000001, 0.01, 2, 4,
       6, tanh(u),
       4, tanh(u),
       3, relu,
       sigmoid);
```

```java
define(4000, 2000, 0.0000001, 0.01, 2, 4,
       12, tanh(u),
       12, tanh(u),
       12, relu,
       sigmoid);
```

```java
define(32, 2000, 0.0000001, 0.01, 2, 4,
       16, tanh(u),
       12, tanh(u),
       8, tanh(u),
       sigmoid);
```

---

### Notes

* Modify only the `define` to change training parameters and network architecture.
* Training and testing are executed automatically from `Perceptron_Classifier`.

---


