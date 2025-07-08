# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the CSV
# df = pd.read_csv("multi_goal_steering.csv")  # Replace with your actual CSV path

# # Replace "not low" with "high" and "not high" with "low" in the label
# df["label"] = df["label"].str.replace("not low", "high", regex=False)
# df["label"] = df["label"].str.replace("not high", "low", regex=False)

# # Set Seaborn theme
# sns.set(style="whitegrid")

# # Create 4 separate subplots (one per label_combo)
# g = sns.FacetGrid(df, col="label", col_wrap=2, height=4, sharey=False)
# g.map_dataframe(sns.lineplot, x="step", y="metric_value", marker="o")

# # Add titles and axis labels
# g.set_titles("{col_name}")
# g.set_axis_labels("Step", "Metric Value")

# # Tidy layout
# plt.tight_layout()

# # Save the full plot to file
# g.savefig("line_plots_by_label_combo.png", dpi=300)

# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load CSV
df = pd.read_csv("multi_goal_steering.csv")

# Replace "not low" and "not high" in labels if needed
df["label"] = df["label"].str.replace("not low", "high", regex=False)
df["label"] = df["label"].str.replace("not high", "low", regex=False)

labels = df["label"].unique()

# Set up 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors for each line
colors = ['red', 'blue', 'green', 'orange']

for color, label in zip(colors, labels):
    subset = df[df["label"] == label].sort_values(by="step")
    x = subset["step"]
    y = subset["charge_ph7"]
    z = subset["gravy_score"]
    
    ax.plot(x, y, z, label=label, color=color, marker='o')

ax.set_xlabel("Step")
ax.set_ylabel("Charge at pH7")
ax.set_zlabel("GRAVY Score")
ax.set_title("Multi-goal Steering: Charge at pH7 & GRAVY over Steps")
ax.legend(loc='upper left')

plt.tight_layout()

# Save the figure
fig.savefig("multi_goal_steering_3d_lines.png", dpi=300)

plt.show()
