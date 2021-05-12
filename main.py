import pandas as pd
import numpy as np
#import keras 

from astropy.io import fits
import aplpy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

def prepareImageSecondMethod():
	image_file = fits.open('../data/raw/SKAMid_B1_1000h_v3.fits')
	image_data = image_file[0]
	image_data = image_data.data.reshape((32768,32768))
	image_data = image_data[:15000,:15000]
	img = aplpy.FITSFigure(image_data)
	img.show_colorscale(cmap='gist_heat')
	img.save('imageResult2.png')

def prepareImageFirstMethod():
	img = aplpy.FITSFigure('../data/raw/SKAMid_B1_1000h_v3.fits', downsample=25)
	img.show_colorscale(cmap='gist_heat')
	img.save('../data/imageResult1.png')


def showImage():
	img = mpimg.imread('imageResult1.png')
	imgplot = plt.imshow(img)
	plt.show()

def showDataSet():
	TrainingSet=pd.read_csv("../data/ground-truth/TrainingSet_B1_v2.txt",skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']

	print(TrainingSet)
	print(TrainingSet.len())

def divideImages():
	# Divide the fits image in 50x50 images
	fits_img = fits.open("../data/raw/SKAMid_B1_1000h_v3.fits")
	TrainingSet=pd.read_csv("../data/ground-truth/TrainingSet_B1_v2.txt",skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']
	TrainingSet['x'] = TrainingSet['x'].astype(int)
	TrainingSet['y'] = TrainingSet['y'].astype(int)
    #print(fits_img.info())
	X_PIXEL_RES = abs(fits_img[0].header['CDELT1'])
	Y_PIXEL_RES = abs(fits_img[0].header['CDELT2'])

	print(fits_img[0].data.shape)

	fits_img=fits_img[0].data[0,0]

	img_array = np.empty((0,64,64))

	for i in range(16300,32768,64):
		for j in range(16300,32768,64):
			img_array = fits_img[i:i+64,j:j+64]
			filter_x = (TrainingSet['x'] < i+64) & (TrainingSet['x'] >= i)
			filter_y = (TrainingSet['y'] < j+64) & (TrainingSet['y'] >= j)
			small_ts = TrainingSet[(filter_x) & (filter_y)]
			if (len(small_ts)) > 0:
				print('CIAO')
				_, ax = plt.subplots()
				ax.imshow(img_array, cmap='gist_heat')
				for _, row in small_ts.iterrows():
					major = row['BMAJ'] / 3600 / X_PIXEL_RES / 2
					minor = row['BMIN'] / 3600 / X_PIXEL_RES / 2
					phi = np.radians(row['PA'])
					x = row['x'] - i
					y = row['y'] - j
					center = patches.Circle((x, y), radius=1, edgecolor='g', facecolor='none')
					xmin, ymin, xmax, ymax = ellipse_to_box(phi, major, minor, x, y)
					box = patches.Rectangle((xmin, ymin), 
											width=xmax - xmin, 
											height=ymax - ymin,
											linewidth=1,
											edgecolor='y',
											facecolor='none')
					ax.add_patch(box)
					ax.add_patch(center)			
				plt.show()
			#np.save("../data/training/B1_1000h/img" + str(i) + str(j) + ".npy", img_array)
			#small_ts.to_csv('../data/training/csv/img' + str(i) + str(j) + ".csv", header=None)
			#print("I: ",i," ",i+64)
			#print("J: ",j," ",j+64)
			#print(small_ts[['ID','x','y']])
			#if (len(small_ts)) >0:
				#plt.imshow(img_array,cmap='gist_heat')
				#plt.show()
				#return
			
def ellipse_to_box(phi, major, minor, x, y):
	axis_ux = major * np.cos(phi)
	axis_uy = major * np.sin(phi)
	axis_vx = minor * np.cos(phi + np.pi / 2)
	axis_vy = minor * np.sin(phi + np.pi / 2)

	box_halfwidth = np.sqrt(axis_ux ** 2 + axis_vx ** 2)
	box_halfheight = np.sqrt(axis_uy ** 2 + axis_vy ** 2)

	xmin, ymin = x - box_halfwidth, y - box_halfheight
	xmax, ymax = x + box_halfwidth, y + box_halfheight

	return (xmin, ymin, xmax, ymax)

divideImages()
#array = np.load("img_array.npy")
#for i in range(0,10):
	# Save the generated images
	#np.save("img_array.npy",img_array)
#	plt.imshow(array[i], cmap='gist_heat')
#	plt.show()