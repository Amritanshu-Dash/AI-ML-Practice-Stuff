import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
  'sqft': [1500, 1600, 1700, 1800, 1900],
  'bedrooms': [3, 3, 4, 4, 5],
  'price_lakhs': [50, 55, 60, 65, 70]
}
df = pd.DataFrame(data)
print("Original DataFrame:", df, sep="\n")

#converting to numpy array
x = df.values ## converting dataframe to numpy array for mathematical operations
x_centered = x - x.mean(axis=0) ## centering the data by subtracting mean of each column from respective column values, why ? to make data centered around origin for better PCA results

#covariance matrix + Eigen
cov = np.cov(x_centered.T) ## calculating covariance matrix of transposed centered data, transposing because np.cov expects features in rows and samples in columns, covariance matrix tells us how features vary together
eigenvalues, eigenvectors = np.linalg.eig(cov) ## calculating eigenvalues and eigenvectors of covariance matrix, eigenvalues tell us the amount of variance captured by each principal component, eigenvectors tell us the direction of these components, bigger eigenvalue means more important component

#project to 2D
proj = x_centered @ eigenvectors[:,:2] ## projecting centered data onto first 2 eigenvectors (principal components) to reduce dimensionality to 2D for visualization

#plot
plt.figure(figsize=(8,6))

plt.scatter(proj[:,0], proj[:, 1], c='blue') ## plotting the projected data points in 2D space
for i, txt in enumerate(df.index + 1): ## annotating each point with its index + 1 for clarity like H1, H2, ...
    plt.annotate(f'H{txt}', (proj[i,0], proj[i,1]))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Housing Data')
plt.grid(True)
plt.show()
