## 📂 Project Structure & Class Overview

| Class                 | Purpose / Description                                                                                                                                                                                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Point**             | Represents a 2D data point. Stores feature coordinates *(x1, x2)* and the assigned cluster label. Provides getters and setters used during clustering iterations.                                                                                                                                                              |
| **Centroid**          | Represents a cluster centroid in 2D space. Stores centroid coordinates and supports relocation during centroid update steps.                                                                                                                                                                                                   |
| **KMeans**            | Implements the K-Means clustering algorithm from scratch. Handles CSV data loading, random centroid initialization, Euclidean distance computation, point-to-cluster assignment, centroid updates, convergence checking, clustering error computation, and optional result export to CSV files.                                |
| **ClusterResearch**   | Acts as an experimental controller for clustering analysis. Executes multiple K-Means runs for different numbers of clusters *(M)*, repeats experiments to mitigate sensitivity to random initialization, tracks the best-performing clustering based on total error, and stores optimal centroids and error metrics to files. |
| **GeneratePointsCSV** | Generates a synthetic two-dimensional dataset for clustering experiments. Creates multiple dense square-shaped regions and uniform background noise, and exports the generated points to a CSV file used as input for the K-Means algorithm.                                                                                   |

---

### Run Instructions

1. Compile all classes:

```
javac *.java
```

2. Generate the dataset:

```
java GeneratePointsCSV
```

3. Run the K-Means clustering program:

```
java KMeans
```

or run clustering experiments across multiple cluster sizes:

```
java ClusterResearch
```

---

### `define` Format

```
define(M)
```

Where:

* **M** → Number of clusters (centroids)

---

### Example `define` Usage

```
define(5)
```

```
define(9)
```

---

### Notes

* Input data is loaded from `points.csv`, generated via `GeneratePointsCSV`.
* The dataset contains multiple dense regions and uniform background noise to simulate realistic clustering challenges.
* Centroids are initialized randomly from existing points.
* The algorithm iterates until centroid displacement falls below a convergence threshold.
* Clustering performance is evaluated using total and average Euclidean distance error.
* `ClusterResearch` performs repeated runs per **M** to avoid poor local minima and identifies the optimal clustering configuration.
