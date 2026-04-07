import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('cache', exist_ok=True)

# Load image di grayscale
img = cv2.imread('images/foto.jpg', 0)

# Tambah salt & pepper noise
noisy = img.copy()
prob = 0.05

# Salt
salt = np.random.rand(*img.shape) < prob/2
# Pepper
pepper = np.random.rand(*img.shape) < prob/2

noisy[salt] = 255
noisy[pepper] = 0

# Restorasi (median filter)
restored = cv2.medianBlur(noisy, 3)
# 3x3 untuk median filter

# Simpan
cv2.imwrite('cache/sp_noisy.png', noisy)
cv2.imwrite('cache/sp_restored.png', restored)

# Tampil
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title('Original'); plt.imshow(img, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title('Salt & Pepper'); plt.imshow(noisy, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title('Restored'); plt.imshow(restored, cmap='gray'); plt.axis('off')
plt.show()