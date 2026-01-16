import matplotlib.pyplot as plt

# Δεδομένα από τα αποτελέσματά σου
M = [3, 5, 7, 9, 11, 13]
errors = [
    570.5767068551371,
    315.719196814037,
    265.00883985274703,
    217.72785125445523,
    202.70013287543483,
    186.37393562690917
]

# Υπολογισμός μέσου σφάλματος (1200 παραδείγματα)
avg_errors = [e / 1200 for e in errors]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(M, errors, marker='o', label="Total Error")

plt.xlabel("Number of Clusters (M)", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.title("Clustering Error vs Number of Clusters", fontsize=14)

plt.grid(True)
plt.legend()
plt.tight_layout()

# Αποθήκευση ή εμφάνιση
plt.savefig("clustering_error_plot.png", dpi=200)
plt.show()
