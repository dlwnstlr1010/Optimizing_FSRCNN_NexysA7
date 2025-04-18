import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("summary_results.csv")

plt.figure(figsize=(10, 6))
plt.scatter(df["param_count"], df["psnr"], c="royalblue", edgecolors="black", alpha=0.75)
plt.title("PSNR vs. Parameter Count (FSRCNN Models)", fontsize=14)
plt.xlabel("Parameter Count", fontsize=12)
plt.ylabel("PSNR", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("results/summary_plot.png")
plt.show()
