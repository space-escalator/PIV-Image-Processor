import cv2
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt

def load_and_grayscale_image(filename):
	img = cv2.imread('PIV_sample_cropped.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def gauss_blur(img):
	return cv2.GaussianBlur(img, (5,5), 0)

def threshold(img):
	return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
		cv2.THRESH_BINARY, 5, -3)

def quick_plot(img):
	plt.figure()
	plt.imshow(img, 'gray', vmin=0, vmax=255, extent=(0,len(img[0]),len(img),0))

def display_thresholding(img):
	blurred_img = gauss_blur(img)
	thresholded_original = threshold(img)
	thresholded_blurred = threshold(blurred_img)

	plt.figure()

	plt.subplot(2,2,1)
	plt.imshow(img, 'gray')
	plt.title('Original Image')
	plt.xticks([]),plt.yticks([])

	plt.subplot(2,2,2)
	plt.imshow(blurred_img, 'gray')
	plt.title('Blurred Image')
	plt.xticks([]),plt.yticks([])

	plt.subplot(2,2,3)
	plt.imshow(thresholded_original, 'gray')
	plt.title('Thresholded Original Image')
	plt.xticks([]),plt.yticks([])

	plt.subplot(2,2,4)
	plt.imshow(thresholded_blurred, 'gray')
	plt.title('Thresholded Blurred Image')
	plt.xticks([]),plt.yticks([])

def isolate_particles(img, label_array, particle_count):
	# img is already thresholded

	particle_images = []

	# print(label_array[len(img)-100:len(img), len(img[0])-100:len(img[0])])

	for i in range(1,particle_count+1):
		coords = np.where(label_array == i)

		# Define cropping window
		min_y = min(coords[0])
		max_y = max(coords[0])
		min_x = min(coords[1])
		max_x = max(coords[1])

		# Create a copy of img that doesn't include other particles in the cropping window
		masked_img = np.copy(img)
		for j in range(min_y,max_y+1):
			for k in range(min_x,max_x+1):
				if label_array[j,k] != i and label_array[j,k] != 0:
					masked_img[j,k] = 0

		cropped_img = masked_img[min_y:max_y+1, min_x:max_x+1]

		if count_pixels(cropped_img) == 0:
			print(((min_x, min_y), (max_x, max_y)))

		particle_images.append(cropped_img)

	return particle_images

def count_pixels(img):
	# Count the number of non-zero pixels in img
	count = 0
	for row in img:
		for pixel in row:
			if pixel != 0:
				count += 1

	return count



if __name__ == "__main__":
	filename = 'PIV_sample_cropped.jpg'
	img = load_and_grayscale_image(filename)

	thresholded_img = threshold(gauss_blur(img))

	label_array, particle_count = ndimage.label(thresholded_img)
	particle_images = isolate_particles(thresholded_img, label_array, particle_count)
	particle_pixel_counts = [count_pixels(i) for i in particle_images]
	
	print('The image contains %d particles' % particle_count)
	print(np.bincount(particle_pixel_counts))
	plt.figure()
	plt.hist(particle_pixel_counts, bins=range(max(particle_pixel_counts)))
	plt.title('Histogram of Pixels per Droplet')
	plt.xlabel('Pixels in One Droplet')

	# display_thresholding(img)
	square_width = 100
	square_start_y = int(len(img)/2-square_width/2)
	square_start_x = int(len(img[0])/2-square_width/2)
	center_block_img = img[square_start_y:square_start_y+square_width, square_start_x:square_start_x+square_width]
	# display_thresholding(center_block_img)
	# plt.show()

	
	current_img = threshold(gauss_blur(center_block_img))
	current_label_array, current_particle_count = ndimage.label(current_img)
	particle_images = isolate_particles(current_img, current_label_array, current_particle_count)

	# Plot all particles in a grid in a single figure
	plt.figure()
	grid_width = np.ceil(np.sqrt(len(particle_images)))

	for i in range(len(particle_images)):
		plt.subplot(grid_width, grid_width, i+1)
		plt.imshow(particle_images[i], 'gray', vmin=0, vmax=255)
		plt.xticks([]),plt.yticks([])

	quick_plot(current_img)

	plt.show()



