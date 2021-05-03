import pandas as pd
import numpy as np
#import keras

from astropy.io import fits
import aplpy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def prepareImageSecondMethod():
	image_file = fits.open('SKAMid_B1_8h_v3.fits')
	image_data = image_file[0]
	image_data = image_data.data.reshape((32768,32768))
	image_data = image_data[:15000,:15000]
	img = aplpy.FITSFigure(image_data)
	img.show_colorscale(cmap='gist_heat')
	img.save('imageResult2.png')

def prepareImageFirstMethod():
	img = aplpy.FITSFigure('SKAMid_B1_8h_v3.fits', downsample=25)
	img.show_colorscale(cmap='gist_heat')
	img.save('imageResult1.png')


def showImage():
	img = mpimg.imread('imageResult1.png')
	imgplot = plt.imshow(img)
	plt.show()

def showDataSet():
	TrainingSet=pd.read_csv("TrainingSet_B1_v2.txt",skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']

	print(TrainingSet)
	print(TrainingSet.len())

def divideImages():
	# Divide the fits image in 50x50 images
	fits_img = fits.open("SKAMid_B1_8h_v3.fits")
	TrainingSet=pd.read_csv("TrainingSet_B1_v2.txt",skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']
	#print(fits_img.info())

	print(fits_img[0].data.shape)

	fits_img=fits_img[0].data[0,0]

	img_array = np.empty((0,64,64))

	for i in range(16300,20300,64):
		for j in range(16300,20300,64):
			img_array = fits_img[i:i+64,j:j+64]
			filter_x = (TrainingSet['x'] < i+64) & (TrainingSet['x'] >= i)
			filter_y = (TrainingSet['y'] < j+64) & (TrainingSet['y'] >= j)
			small_ts = TrainingSet[(filter_x) & (filter_y)]
			print("I: ",i," ",i+64)
			print("J: ",j," ",j+64)
			print(small_ts[['x','y']])
			if (len(small_ts)) >0:
				plt.imshow(img_array,cmap='gist_heat')
				plt.show()
				return
			

	

divideImages()
#array = np.load("img_array.npy")
#for i in range(0,10):
	# Save the generated images
	#np.save("img_array.npy",img_array)
#	plt.imshow(array[i], cmap='gist_heat')
#	plt.show()