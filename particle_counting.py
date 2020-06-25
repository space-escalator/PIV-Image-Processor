"""
This is a small library of functions to count the particles in, and return statistics about, 
PIV image particles, given an image.

TODO: References
"""

import cv2
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt

__author__ = "Aaron Goldfogel"
__email__ = "aaron@goldfogel.space"

def load_and_grayscale_image(filename):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def load_and_grayscale_data(filename):
	datalist = []
	with open(filename) as reader:
		in_header = True
		for line in reader:
			split_line = line.split()

			# skip headers
			if in_header:
				in_header = False
				if not all([str(word).strip('-').replace('.','').isnumeric() for word in split_line]):
					in_header = True

			if not in_header:
				if not all([str(word).strip('-').replace('.','').isnumeric() for word in split_line]):
					break
					# This is a hack to ignore the second ZONE and I don't like it
				# read every line into the list
				datalist.append([float(i) for i in split_line])

	pixel_width = datalist[1][0] - datalist[0][0] # Assumes pixels are square

	number_of_columns = 0
	for i in range(len(datalist)):
		if datalist[i+1][1] < datalist[i][1]:
			number_of_columns = i+1
			break

	number_of_rows = int(len(datalist) / number_of_columns)

	img = np.zeros((number_of_rows, number_of_columns))
	k = 0
	for i in range(len(img)):
		for j in range(len(img[i,:])):
			img[i,j] = datalist[k][2]
			k += 1

	# Drop the floor to zero and mask anything above 255 to 255 to not lose low-level precision
	img -= img.min()
	# img /= (img.max()/255)
	for i in range(len(img)):
		for j in range(len(img[i])):
			if img[i,j] > 255:
				img[i,j] = 255

	img = img.astype('uint8')

	return img, pixel_width

def gauss_blur(img):
	return cv2.GaussianBlur(img, (3,3), 0)

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

def find_external_corners(img):
	# Given a binarized image, return a list of coordinates of all the exterior corner coordinates
	external_corners = [] # Variable to return

	lower_border = len(img)
	right_border = len(img[0])

	# Iterate through every vertex in the image and add it to the list if it only borders 1 white pixel
	for y in range(lower_border+1):
		for x in range(right_border+1):
			# The vertex is at (y,x). (y,x) is also the pixel below and to the right of th vertex.
			adjacent_white = 0
			
			# up left
			if y != 0 and x != 0:
				if img[y-1,x-1] != 0:
					adjacent_white += 1
			# up right
			if y != 0 and x != right_border:
				if img[y-1,x] != 0:
					adjacent_white += 1
			# down left
			if y != lower_border and x != 0:
				if img[y,x-1] != 0:
					adjacent_white += 1
			# down right
			if y != lower_border and x != right_border:
				if img[y,x] != 0:
					adjacent_white += 1

			if adjacent_white == 1:
				external_corners.append([y,x])

	return external_corners

def plot_external_corners(img):
	# Plot the external corners of the image as red crosses on top of the image
	plt.imshow(img, 'gray', vmin=0, vmax=255, extent=(0,len(img[0]),len(img),0))
	
	corners = find_external_corners(img)
	x = [i[1] for i in corners]
	y = [i[0] for i in corners]

	plt.scatter(x,y, c='r', marker='P', s=10)

def count_to_diameter(count, pixel_width):
	"""
	Returns the diameter of a circle with the same area 
	as [count] squares that are [pixel_width] wide
	"""
	area = count*(pixel_width**2)
	return np.sqrt(area/np.pi)*2




if __name__ == '__main__':
	filename = 'images/B00001.dat'
	pixel_length_unit = 'mm'
	# img = load_and_grayscale_image(filename) # This is for if the image is already an image
	# pixel_width = 0.0001
	img, pixel_width = load_and_grayscale_data(filename) # This is for if the image is a .dat file of xyz data

	thresholded_img = threshold(gauss_blur(img))

	label_array, particle_count = ndimage.label(thresholded_img)
	particle_images = isolate_particles(thresholded_img, label_array, particle_count)
	particle_pixel_counts = [count_pixels(i) for i in particle_images]
	particle_diameters = [count_to_diameter(i, pixel_width) for i in particle_pixel_counts]
	
	unique_diameters = sorted(set(particle_diameters))
	bin_width = max(unique_diameters)
	for i in range(len(unique_diameters)-1):
		bin_diff = unique_diameters[i+1]-unique_diameters[i]
		if bin_width > bin_diff: bin_width = bin_diff
	
	bins = np.arange(0,max(unique_diameters),bin_width)

	print('The image contains %d particles' % particle_count)
	print(np.bincount(particle_pixel_counts))
	
	plt.figure()
	plt.hist(particle_diameters, bins=bins, rwidth=.1)
	plt.title('Particle Diameter Distribution | Total Particles: %d' %particle_count)
	plt.xlabel('Diameter (%s)' %pixel_length_unit)
	locs, labels = plt.yticks()
	labels = ['%1.3f' %(float(i)/particle_count) for i in locs]
	plt.yticks(locs, labels)
	plt.ylabel('Normalized Particle Count')


	plt.figure()
	plt.hist(particle_diameters, bins=10)
	plt.title('Particle Diameter Distribution | Total Particles: %d' %particle_count)
	plt.xlabel('Diameter (%s)' %pixel_length_unit)
	locs, labels = plt.yticks()
	labels = ['%1.3f' %(float(i)/particle_count) for i in locs]
	plt.yticks(locs, labels)
	plt.ylabel('Normalized Particle Count')

	display_thresholding(img)
	square_width = 20
	square_start_y = int(len(img)/2-square_width/2)
	square_start_x = int(len(img[0])/2-square_width/2)
	center_block_img = img[square_start_y:square_start_y+square_width, square_start_x:square_start_x+square_width]
	display_thresholding(center_block_img)

	
	current_img = threshold(gauss_blur(center_block_img))
	current_label_array, current_particle_count = ndimage.label(current_img)
	particle_images = isolate_particles(current_img, current_label_array, current_particle_count)

	# Plot all particles in a grid in a single figure
	plt.figure()
	grid_width = np.ceil(np.sqrt(len(particle_images)))

	for i in range(len(particle_images)):
		plt.subplot(grid_width, grid_width, i+1)
		# plt.imshow(particle_images[i], 'gray', vmin=0, vmax=255)
		plot_external_corners(particle_images[i])
		plt.xticks([]),plt.yticks([])

	quick_plot(current_img)

	plt.show()



