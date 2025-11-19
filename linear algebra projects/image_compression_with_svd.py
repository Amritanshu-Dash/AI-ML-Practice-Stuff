import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = '/Users/amritanshudash/Downloads/x7.jpg'#path to image

img = Image.open(img_path).convert('L') #convert to grayscale
img_matrix = np.array(img, dtype=float) #convert image to numpy array
print("Original Image shape:", img_matrix.shape)

#SVD
U, s, Vt = np.linalg.svd(img_matrix, full_matrices=False) #perform SVD

#Change k for compression level like try 5, 20, 50, 100
k = 50
img_matrix_compressed = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:] #reconstruct image using top k singular values/vectors

#Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_matrix, cmap='gray')
plt.title(f'Original Image: ({img_matrix.shape[0]} * {img_matrix.shape[1]})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_matrix_compressed, cmap='gray')
plt.title(f'Compressed Image with k={k}')
plt.axis('off')

#compression ratio
original_size = img_matrix.size
compressed_size = k * (img_matrix.shape[0] + img_matrix.shape[1] + 1) # U (m*k) + S (k) + Vt (k*n)
print(f"original size: {original_size}")
print(f"compressed size: {compressed_size}")
compression_ratio = original_size / compressed_size

plt.suptitle("SVD Image Compression")
plt.show()
