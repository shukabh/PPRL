import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create the DataFrame with the additional 'Total' row
data = {
    'State': ['nsw', 'other', 'qld', 'sa', 'vic', 'wa', 'Total'],
    'Count_rl': [1643, 448, 864, 368, 1205, 472, 5000],
    'Count_private': [1597.01, 431.00, 878.00, 358.00, 1179.01, 445.00, 4888.02],
    'True_Count': [1637, 445, 897, 371, 1192, 458, 5000]
}

df = pd.DataFrame(data)

# Calculate differences
df['Diff_rl'] = df['Count_rl'] - df['True_Count']
df['Diff_private'] = df['Count_private'] - df['True_Count']

# Plotting
x = np.arange(len(df['State']))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, df['Diff_rl'], width, label='Count_rl - True')
bars2 = ax.bar(x + width/2, df['Diff_private'], width, label='Count_private - True')

# Add labels and title
ax.set_xlabel('State')
ax.set_ylabel('Difference from True Count')
ax.set_title('Difference Between Estimates and True Count by State (Including Total)')
ax.set_xticks(x)
ax.set_xticklabels(df['State'])
ax.axhline(0, color='gray', linewidth=0.8)
ax.legend()

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.show()
