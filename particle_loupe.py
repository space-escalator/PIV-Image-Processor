import cv2
import numpy as np
from scipy import ndimage
import particle_counting
from particle_batch_processor import sort_and_deduplicate

from matplotlib import pyplot as plt

class Loupe(object):
	def __init__(self, img, pixelLength = 1, pixelLengthUnit = 'um'):
		self._img = img
		self._windowManager = WindowManager('Loupe', 
			self.onKeypress, 
			self.onBlurTrackbar, 
			self.onHistogramTrackbar, 
			self.onBlockSizeTrackbar, 
			self.onThresholdConstantTrackbar)
		self._magX = 0
		self._magY = 0
		self._magHeight = 100
		self._filtersOn = False
		self._pixelLength = pixelLength
		self._pixelLengthUnit = pixelLengthUnit
		self._blurKernelSize = 3
		self._enableStatistics = False
		self._thresholdBlockSize = 7
		self._thresholdConstant = -3

		plt.figure() # initialize a figure for plotting histograms
		plt.ion() # turn on interactivity

	def run(self):
		"""
		Main loop.
		"""
		self._windowManager.createWindow()
		while self._windowManager.isWindowCreated:
			img1 = self._img.copy()
			
			if self._filtersOn:
				img1 = self.filterImage(img1)
				
				if self._statisticsObsolete and self._enableStatistics:
					# This is slow
					self.runStatistics(img1)
			
			img2 = self.magnify(img1, self._magX, self._magY)

			# Convert from Grayscale to BGR
			img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
			img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
			
			# Draw green magnification indicators
			img1 = cv2.rectangle(img1, (self._magX, self._magY), 
				(self._magX+self._magHeight, self._magY+self._magHeight),
				color = (0,255,0), thickness = 2)
			img1 = cv2.line(img1,
				(self._magX+self._magHeight, self._magY),
				(self._img.shape[1],0),
				color = (0,255,0), thickness = 2)
			img1 = cv2.line(img1,
				(self._magX+self._magHeight, self._magY+self._magHeight),
				(self._img.shape[1], self._img.shape[0]),
				color = (0,255,0), thickness = 2)

			frame = np.hstack((img1, img2))
			self._windowManager.show(frame)
			self._windowManager.processEvents()

	def onKeypress(self, keycode):
		"""Process keypress events"""
		step_amount = 20

		if keycode == 27: # esc
			self._windowManager.destroyWindow()
		elif keycode == 32: # space
			if self._filtersOn:
				self._filtersOn = False
			else:
				self._filtersOn = True
				self._statisticsObsolete = True
		elif keycode == 97: # a
			if self._magX >= step_amount:
				self._magX -= step_amount
			else:
				self._magX = 0
		elif keycode == 119: # w
			if self._magY >= step_amount:
				self._magY -= step_amount
			else:
				self._magY = 0
		elif keycode == 100: # d
			if self._magX <= self._img.shape[1] - step_amount - self._magHeight:
				self._magX += step_amount
			else:
				self._magX = self._img.shape[1] - self._magHeight
		elif keycode == 115: # s
			if self._magY <= self._img.shape[0] - step_amount - self._magHeight:
				self._magY += step_amount
			else:
				self._magY = self._img.shape[0] - self._magHeight
		elif keycode == 114: # r
			pass # just refresh

	def filterImage(self, img):
		"""
		Apply a Gaussian Blur to the image, then threshold it using a Gaussian
		adaptive threshold. 3 Important vairiables for the blur and thresholding
		are passed in. The output image has only pixel values of 0 and 255.
		Pretty much identical to particle_batch_processor.filterImage(), but uses 
		the Loupe object's local variables.

		TODO: just use particle_batch_processor.filterImage()
		"""
		img2 = img.copy()
		img2 = cv2.GaussianBlur(img2, (self._blurKernelSize,self._blurKernelSize), 0)
		img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
			cv2.THRESH_BINARY, self._thresholdBlockSize, self._thresholdConstant)
		return img2

	def runStatistics(self, thresholded_img):
		"""
		Run particle counting statistics on an image that is
		already thresholded.

		TODO: Separate statistics math and plotting into separate functions
		TODO: Just use the stuff from particle_batch_processor.py to avoid
		duplicated code.
		"""
		label_array, particle_count = ndimage.label(thresholded_img)
		particle_pixel_counts = particle_counting.count_unique_occurrances(label_array)

		unique_pixel_counts = sort_and_deduplicate(particle_pixel_counts)
		bin_width = max(unique_pixel_counts)
		for i in range(len(unique_pixel_counts)-1):
			bin_diff = unique_pixel_counts[i+1]-unique_pixel_counts[i]
			if bin_width > bin_diff: bin_width = bin_diff

		bins = np.arange(0,max(unique_pixel_counts),bin_width)

		# Convert bins and particle_pixel_counts to diameters, not pixel counts
		bins = np.array([particle_counting.count_to_diameter(count, self._pixelLength) for count in bins])
		particle_diameters = [particle_counting.count_to_diameter(count, self._pixelLength) for count in particle_pixel_counts]

		particle_count = len(particle_diameters)

		plt.clf() # Clear whatever might have been in the figure
		plt.hist(particle_diameters, bins=bins, rwidth=.1)
		plt.title('Particle Diameter Distribution | Total Particles: %d' %particle_count)
		plt.xlabel('Diameter (%s)' %self._pixelLengthUnit)
		locs, labels = plt.yticks()
		labels = ['%1.3f' %(float(i)/particle_count) for i in locs]
		plt.yticks(locs, labels)
		plt.ylabel('Normalized Particle Count')

		plt.show(block=False)
		self._statisticsObsolete = False

	def magnify(self, img, magX, magY):
		"""
		Take an image and magnify a self._magHeight*self._magHeight square
		beginning at magX, magY.
		Returns the magnified sub-image at vertical resolution of the original
		image. Uses nearest-neighbor interpolation to avoid blurring. For this reason,
		self._magHeight must be a factor of the vertical resolution of the 
		original image.
		"""
		imgHeight = self._img.shape[0] # Not sure why this uses self._img not just img
		while imgHeight % self._magHeight != 0:
			self._magHeight -= 1

		magnified = img[magY:magY+self._magHeight, magX:magX+self._magHeight]
		magnified = cv2.resize(magnified, (imgHeight, imgHeight), 
			interpolation = cv2.INTER_NEAREST)
		return magnified

	# Trackbar math and actions. Convert trackbar values to variable values.
	def onBlurTrackbar(self, value):
		self._blurKernelSize = 2 * value + 1
		self._statisticsObsolete = True

	def onHistogramTrackbar(self, value):
		self._enableStatistics = bool(value)

	def onBlockSizeTrackbar(self, value):
		self._thresholdBlockSize = 1 + 2*value

	def onThresholdConstantTrackbar(self, value):
		self._thresholdConstant = value - 10

	def exportSettings(self):
		return self._blurKernelSize, self._thresholdBlockSize, self._thresholdConstant

class WindowManager(object):
	def __init__(self, windowName, 
		keypressCallback = None, 
		blurCallback = None,
		statisticsCallback = None,
		thresholdBlockCallback = None,
		thresholdConstantCallback = None):
		
		self.keypressCallback = keypressCallback
		self.blurCallback = blurCallback
		self.statisticsCallback = statisticsCallback
		self.thresholdBlockCallback = thresholdBlockCallback
		self.thresholdConstantCallback = thresholdConstantCallback
		self._windowName = windowName
		self._isWindowCreated = False

	@property
	def isWindowCreated(self):
		return self._isWindowCreated
	def createWindow(self):
		cv2.namedWindow(self._windowName)
		cv2.createTrackbar('Histogram (OFF/ON)', self._windowName, 0, 1, self.statisticsCallback)
		cv2.createTrackbar('Blur Kernel Size', self._windowName, 1, 4, self.blurCallback)
		cv2.createTrackbar('Threshold Block Size', self._windowName, 3, 10, self.thresholdBlockCallback)
		cv2.createTrackbar('Threshold Constant', self._windowName, 7, 20, self.thresholdConstantCallback)
		self._isWindowCreated = True
	def show(self, frame):
		cv2.imshow(self._windowName, frame)
	def destroyWindow(self):
		cv2.destroyWindow(self._windowName)
		self._isWindowCreated = False
	def processEvents(self):
		keycode = cv2.waitKey(0)
		if self.keypressCallback is not None and keycode != -1:
			self.keypressCallback(keycode)


if __name__ == '__main__':
	filename = 'images/B00001.dat'
	# img = cv2.pyrDown(cv2.imread('images/Main Parachute Rainier Vista.jpeg', 1))
	img, pixelLength = particle_counting.load_and_grayscale_data(filename)
	Loupe(img, pixelLength*1000, 'um').run()