import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import wiener

os.makedirs('cache', exist_ok=True)

# Load image
img = cv2.imread('images/foto.jpg', 0)

if img is None:
    print("File tidak ditemukan")
    exit()

# PSF
psf = np.ones((3,3)) / 9

# Blur
blurred = cv2.filter2D(img, -1, psf)

# Wiener filtering
restored = wiener(blurred / 255.0, psf, balance=0.1)
restored = np.clip(restored * 255, 0, 255).astype('uint8')

# Normalisasi
restored = restored - restored.min()
restored = restored / restored.max() * 255
restored = restored.astype('uint8')

# Save
cv2.imwrite('cache/wiener_blur.png', blurred)
cv2.imwrite('cache/wiener_result.png', restored)

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
plt.title('Wiener Filter')
plt.imshow(restored, cmap='gray')
plt.axis('off')

plt.show()