import pandas as pd 
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# load_digits() gives 1,797 handwritten digit images (0-9), each an 8x8
# grayscale scan (64 pixel values, 0-16 intensity), stored as flat rows
digits = load_digits()

# Convert this to a pandas dataframe for easier manipulation
df = pd.DataFrame(data=digits.data)
df['label'] = digits.target # Rename the targets column to 'label'

# Drop every row except the labels we care about comparing: 3 and 8
df = df[df['label'].isin([3, 8])]

# Each row is a flattened 8x8 image (64 numbers) — reshape back to 8x8 to plot it
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for ax, idx in zip(axes.flat, df.index[:10]):
    row = df.loc[idx] # Extract the row
    ax.imshow(row.iloc[:64].values.reshape(8, 8), cmap='gray_r') # Show the plots
    ax.set_title(f"label: {row['label']}")
    ax.axis('off')
plt.tight_layout()
plt.show()
