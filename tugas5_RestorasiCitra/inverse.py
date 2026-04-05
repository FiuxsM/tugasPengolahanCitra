import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('cache', exist_ok=True)

# Load image
img = cv2.imread('images/foto.jpg', 0)

if img is None:
    print("File tidak ditemukan")
    exit()

# PSF (blur kernel)
psf = np.ones((3,3)) / 9

# Blur image
blurred = cv2.filter2D(img, -1, psf)

# FFT
G = np.fft.fft2(blurred)
H = np.fft.fft2(psf, s=img.shape)

# Inverse filtering (stabil)
epsilon = 1e-1
F_hat = G * np.conj(H) / (np.abs(H)**2 + epsilon)

restored = np.real(np.fft.ifft2(F_hat))

# Normalisasi
restored = restored - restored.min()
restored = restored / restored.max() * 255
restored = restored.astype('uint8')

# Save
cv2.imwrite('cache/inverse_blur.png', blurred)
cv2.imwrite('cache/inverse_result.png', restored)

# Display
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Blurred')
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Inverse Filter')
plt.imshow(restored, cmap='gray')
plt.axis('off')

plt.show()