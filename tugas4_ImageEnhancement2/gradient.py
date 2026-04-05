import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('cache', exist_ok=True)

img = cv2.imread('images/foto.jpg', 0)

if img is None:
    print("File tidak ditemukan")
    exit()

# 1. Blur (sesuai slide)
blur = cv2.GaussianBlur(img, (0, 0), 1.2)

# 2. Sobel gradient
gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

# 3. Magnitude gradient
grad = cv2.magnitude(gx, gy)

# 4. Sharpen (tambahkan grad ke image)
out = np.clip(blur + 0.7 * grad, 0, 255).astype('uint8')

# Simpan
cv2.imwrite('cache/blur.png', blur)
cv2.imwrite('cache/gradient.png', grad)
cv2.imwrite('cache/sharpen.png', out)

# Tampilkan
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title('Input Blur')
plt.imshow(blur, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Magnitude Gradient')
plt.imshow(grad, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Hasil Sharpen')
plt.imshow(out, cmap='gray')
plt.axis('off')

plt.show()