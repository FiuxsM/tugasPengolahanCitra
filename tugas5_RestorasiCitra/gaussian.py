import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('cache', exist_ok=True)

img = cv2.imread('images/foto.jpg', 0)

# Tambah Gaussian noise
noise = np.random.normal(0, 25, img.shape)
noisy = img + noise
noisy = np.clip(noisy, 0, 255).astype('uint8')

# Restorasi (Gaussian smoothing)
restored = cv2.GaussianBlur(noisy, (5,5), 1.0)

# Simpan
cv2.imwrite('cache/gaussian_noisy.png', noisy)
cv2.imwrite('cache/gaussian_restored.png', restored)

# Tampil
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title('Original'); plt.imshow(img, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title('Gaussian Noise'); plt.imshow(noisy, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title('Restored'); plt.imshow(restored, cmap='gray'); plt.axis('off')
plt.show()