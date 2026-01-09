import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

if 'actual' not in df.columns:
    print("No actual values in CSV!")
    exit()

plt.figure(figsize=(10, 6))
plt.plot(df['id'], df['predicted'], label='Predicted', color='blue')
plt.plot(df['id'], df['actual'], label='Actual', color='orange')

plt.xlabel("ID")
plt.ylabel("House Price")  
plt.title("Predicted vs Actual over samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

if 'actual' not in df.columns:
    print("No actual values in CSV!")
    exit()

# Scatter plot: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(df['actual'], df['predicted'], color='blue', alpha=0.6, label='Predicted')

max_val = max(df['actual'].max(), df['predicted'].max())
plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal prediction')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
