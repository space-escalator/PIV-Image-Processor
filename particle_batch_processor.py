import cv2
import numpy as np
from scipy import ndimage
import particle_counting
import particle_loupe
import os

if __name__ == '__main__':
	directory = 'batch_images/'
	filenames = os.listdir(directory)
	i = len(filenames)-1
	while i >= 0:
		if filenames[i][0] == '.':
			del filenames[i]
		i-=1

	print(filenames)
	
	testImg, pixelLength = particle_counting.load_and_grayscale_data(directory+filenames[0])
	testLoupe = particle_loupe.Loupe(testImg, pixelLength*1000, 'um')
	testLoupe.run()

	settings = testLoupe.export_settings()