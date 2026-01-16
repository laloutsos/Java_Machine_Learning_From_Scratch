import matplotlib.pyplot as plt
import pandas as pd

M = 13

points_file = f'points.csv'
centroids_file = f'best_centroids_M{M}.csv'

points = pd.read_csv(points_file)
centroids = pd.read_csv(centroids_file)

plt.figure(figsize=(6,6))
plt.title(f'Clusters for M={M}')

plt.scatter(points['x1'], points['x2'], marker='+', color='blue', label='Points')

plt.scatter(centroids['x1'], centroids['x2'], marker='*', color='red', s=200, label='Centroids')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
