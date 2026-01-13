import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv('all_models.csv')

exp = data['Experiment']
acc = data['Accuracy']

layerStr = data['Layers']
actStr   = data['Activations']

numModels = len(exp)

# -----------------------------
# Parse layers into numeric arrays
# -----------------------------
totalNeurons = np.zeros(numModels)
numHiddenLayers = np.zeros(numModels)

for i, s in enumerate(layerStr):
    L = eval(s)  # convert string like "[38, 42, 51]" -> list
    totalNeurons[i] = sum(L)
    numHiddenLayers[i] = len(L)

# -----------------------------
# Parse activations into counts
# -----------------------------
activationTypes = ['relu', 'tanh', 'sigmoid']
activationCounts = np.zeros((numModels, len(activationTypes)))

for i, s in enumerate(actStr):
    s_clean = s.replace('[','').replace(']','')
    parts = [x.strip() for x in s_clean.split(',')]
    for p in parts:
        if p.lower() in activationTypes:
            idx = activationTypes.index(p.lower())
            activationCounts[i, idx] += 1

# -----------------------------
# 1) Accuracy per Experiment
# -----------------------------
plt.figure()
plt.plot(exp, acc, '-o', linewidth=1.8, markersize=6)
plt.grid(True)
plt.xlabel('Experiment')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Trained Model')
plt.savefig('plot_accuracy_per_experiment.png')

# -----------------------------
# 2) Scatter Accuracy
# -----------------------------
plt.figure()
plt.scatter(exp, acc, s=60)
plt.grid(True)
plt.xlabel('Experiment')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Distribution')
plt.savefig('plot_scatter_accuracy.png')

# -----------------------------
# 3) Mark Best Model
# -----------------------------
best_idx = acc.idxmax()
bestAcc = acc[best_idx]

plt.figure()
plt.plot(exp, acc, '-o', linewidth=1.8)
plt.plot(exp[best_idx], bestAcc, 'rp', markersize=18, linewidth=2)
plt.grid(True)
plt.xlabel('Experiment')
plt.ylabel('Accuracy (%)')
plt.title('Best Model Highlighted')
plt.legend(['All Models', 'Best Model'])
plt.savefig('plot_best_model.png')

# -----------------------------
# 4) Learning Rate vs Accuracy
# -----------------------------
lr = data['LearningRate']
plt.figure()
plt.scatter(lr, acc, s=60)
plt.xscale('log')
plt.grid(True)
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Accuracy (%)')
plt.title('Learning Rate vs Accuracy')
plt.savefig('plot_lr_vs_accuracy.png')

# -----------------------------
# 5) Epochs vs Accuracy
# -----------------------------
epochs = data['Epochs']
plt.figure()
plt.scatter(epochs, acc, s=60)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Epoch Count vs Accuracy')
plt.savefig('plot_epochs_vs_accuracy.png')

# -----------------------------
# 6) Batch Size vs Accuracy
# -----------------------------
batch = data['Batch']
plt.figure()
plt.scatter(batch, acc, s=60)
plt.grid(True)
plt.xlabel('Batch Size')
plt.ylabel('Accuracy (%)')
plt.title('Batch Size vs Accuracy')
plt.savefig('plot_batch_vs_accuracy.png')

# -----------------------------
# 7) Total Neurons vs Accuracy
# -----------------------------
plt.figure()
plt.scatter(totalNeurons, acc, s=60)
plt.grid(True)
plt.xlabel('Total Neurons Across All Hidden Layers')
plt.ylabel('Accuracy (%)')
plt.title('Model Capacity vs Accuracy')
plt.savefig('plot_total_neurons_vs_accuracy.png')

# -----------------------------
# 8) Number of Hidden Layers vs Accuracy
# -----------------------------
plt.figure()
plt.scatter(numHiddenLayers, acc, s=60)
plt.grid(True)
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy (%)')
plt.title('Depth vs Accuracy')
plt.savefig('plot_layers_vs_accuracy.png')

# -----------------------------
# 9) Heatmap: Activation Composition vs Accuracy
# -----------------------------
plt.figure()
plt.imshow(activationCounts.T, aspect='auto', cmap='viridis')
plt.colorbar()
plt.yticks(range(len(activationTypes)), activationTypes)
plt.xlabel('Experiment')
plt.ylabel('Activation Function Type')
plt.title('Activation Function Usage Distribution')
plt.savefig('plot_activation_heatmap.png')

# -----------------------------
# 10) 3D Feature Map
# -----------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(totalNeurons, lr, acc, c=acc, s=80, cmap='viridis')
ax.set_xlabel('Total Neurons')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Accuracy (%)')
plt.title('3D Feature Space: Capacity vs LR vs Accuracy')
fig.colorbar(p)
plt.savefig('plot_3d_feature_map.png')
