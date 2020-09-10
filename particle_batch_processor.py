import cv2
import numpy as np
from scipy import ndimage
import particle_counting
import particle_loupe
import os

from tqdm import tqdm

from matplotlib import pyplot as plt

def filterImage(img, blurKernelSize, thresholdBlockSize, thresholdConstant):
	"""
	Apply a Gaussian Blur to the image, then threshold it using a Gaussian
	adaptive threshold. 3 Important vairiables for the blur and thresholding
	are passed in. The output image has only pixel values of 0 and 255.
	"""
	img2 = img.copy()
	img2 = cv2.GaussianBlur(img2, (blurKernelSize, blurKernelSize), 0)
	img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
		cv2.THRESH_BINARY, thresholdBlockSize, thresholdConstant)
	return img2

def getParticlePixelCounts(thresholded_img, pixelLength):
	"""
	Returns an unsorted list of the number of pixels in each particle
	in an image that has already been thresholded.
	"""
	label_array, particle_count = ndimage.label(thresholded_img)
	particle_pixel_counts = particle_counting.count_unique_occurrances(label_array)

	return particle_pixel_counts

def uniq(lst):
	"""
	required function for sort_and_duplicate()
	"""
	last = object()
	for item in lst:
		if item == last:
			continue
		yield item
		last = item

def sort_and_deduplicate(l):
	return list(uniq(sorted(l)))

def runStatistics(particle_pixel_counts, pixel_width, length_unit):
	"""
	Prints and plots statistics on an unsorted list of
	pixel counts in each particle.
	"""
	unique_pixel_counts = sort_and_deduplicate(particle_pixel_counts)
	bin_width = max(unique_pixel_counts)
	for i in range(len(unique_pixel_counts)-1):
		bin_diff = unique_pixel_counts[i+1]-unique_pixel_counts[i]
		if bin_width > bin_diff: bin_width = bin_diff

	bins = np.arange(0,max(unique_pixel_counts),bin_width)

	# Convert bins and particle_pixel_counts to diameters, not pixel counts
	bins = np.array([particle_counting.count_to_diameter(count, pixel_width) for count in bins])
	particle_diameters = [particle_counting.count_to_diameter(count, pixel_width) for count in particle_pixel_counts]

	particle_count = len(particle_diameters)

	plt.figure()
	plt.hist(particle_diameters, bins=bins, rwidth=.1)
	plt.title('Particle Diameter Distribution | Total Particles: %d' %particle_count)
	plt.xlabel('Diameter (%s)' %length_unit)
	locs, labels = plt.yticks()
	labels = ['%1.3f' %(float(i)/particle_count) for i in locs]
	plt.yticks(locs, labels)
	plt.ylabel('Normalized Particle Count')

	plt.show(block=True)

if __name__ == '__main__':
	directory = 'batch_images/'
	length_unit = 'um'
	filenames = os.listdir(directory)

	# Get rid of hidden files like .DS_Store
	i = len(filenames)-1
	while i >= 0:
		if filenames[i][0] == '.':
			del filenames[i]
		i-=1
	
	# Run particle_loupe to get desired settings for the batch.
	# Note, this uses the first file, alphabetically.
	# TODO: Make this smarter and optional.
	testImg, pixelLength = particle_counting.load_and_grayscale_data(directory+filenames[0])
	pixelLength *= 1000 #convert from mm to um
	testLoupe = particle_loupe.Loupe(testImg, pixelLength, length_unit)
	testLoupe.run()

	settings = testLoupe.exportSettings()

	# Load, filter, and take statistics on all of the images in the batch.
	# original_images is a tuple of the numpy image and its pixelLength,
	# which is needed for getParticleDiameters.
	print('Loading image files:')
	original_images = []
	for i in tqdm(range(len(filenames))):
		original_images.append(particle_counting.load_and_grayscale_data(directory+filenames[i]))
	# original_images = [particle_counting.load_and_grayscale_data(directory+filename) for filename in filenames]
	print('Filtering and thresholding images...')
	filtered_images = [filterImage(img[0], settings[0], settings[1], settings[2]) for img in original_images]
	
	particle_pixel_counts_big_list = []
	print('Counting particles in each image:')
	for i in tqdm(range(len(filtered_images))):
		particle_pixel_counts_big_list.extend(getParticlePixelCounts(filtered_images[i], original_images[i][1]))

	print('Running statistics...')
	runStatistics(particle_pixel_counts_big_list, pixelLength, length_unit)

	# Remember to use block=True on plt.show()