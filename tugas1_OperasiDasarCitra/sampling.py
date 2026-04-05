import cv2
import matplotlib.pyplot as plt
def run_sampling():
	print("Menjalankan Sampling...")
	# Load gambar
	img = cv2.imread('images/foto.jpg')
	if img is None:
		print("Gambar tidak ditemukan") 
		return
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# Downsampling
	img_small = cv2.resize(img_rgb, (256, 256))
	# Output
	plt.figure(figsize=(8,4))
	plt.subplot(1,2,1)
	plt.title("Original")
	plt.imshow(img_rgb)
	plt.axis("off")
	plt.subplot(1,2,2)
	plt.title("After Sampling (256x256)")
	plt.imshow(img_small)
	plt.axis("off")
	plt.tight_layout()
	plt.show()