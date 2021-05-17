import pandas as pd
import numpy as np
#import keras 

from astropy.io import fits
import astropy.wcs as pywcs
from astropy.nddata.utils import Cutout2D
import aplpy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

IMG_SIZE = 200

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
	fits_img = make_fits_2D(fits_img[0])
	TrainingSet=pd.read_csv("../data/ground-truth/TrainingSet_B1_v2.txt",skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']
	TrainingSet['x'] = TrainingSet['x'].astype(int)
	TrainingSet['y'] = TrainingSet['y'].astype(int)
    #print(fits_img.info())
	X_PIXEL_RES = abs(fits_img.header['CDELT1'])
	WORLD_REF = pywcs.WCS(fits_img.header).deepcopy()

	print(fits_img.data.shape)

	fits_img=fits_img.data[0,0]
	print(fits_img.shape)

	#img_array = np.empty((0,IMG_SIZE,IMG_SIZE))

	for i in range(16300,32768,IMG_SIZE):
		for j in range(16300,32768,IMG_SIZE):
			#img_array = fits_img[i:i+IMG_SIZE,j:j+IMG_SIZE]
			pos = (i + IMG_SIZE/2, j + IMG_SIZE/2)
			img_fits = Cutout2D(fits_img, position=pos, size=IMG_SIZE, wcs=WORLD_REF, copy=True)
			img_array = img_fits.data
			small_ts = TrainingSet.query('x < @i+@IMG_SIZE and x >= @i and y < @j+@IMG_SIZE and y >= @j')
			if (len(small_ts)) > 0:
				print('CIAO')
				#hdu = fits.PrimaryHDU(img_array, header = img_fits.wcs.to_header())
				#hdul = fits.HDUList([hdu])
				#hdul.writeto('new_image.fits')
				_, ax = plt.subplots()
				ax.imshow(img_array, cmap='gist_heat')
				for _, row in small_ts.iterrows():
					major = (row['BMAJ'] / 3600 / X_PIXEL_RES ) / 2
					minor = (row['BMIN'] / 3600 / X_PIXEL_RES ) / 2
					phi = np.radians(row['PA'])
					y = row['x'] % IMG_SIZE
					x = row['y'] % IMG_SIZE
					print('y: ', y, ' x: ', x)
					#ra_min, dec_min, ra_max, dec_max = ellipse_to_box(phi, major, minor, row['RA (centroid)'], row['DEC (centroid)'])
					#xmin, ymin = WORLD_REF.wcs_world2pix([[ra_min, dec_min]], 0)[0]
					#xmax, ymax = WORLD_REF.wcs_world2pix([[ra_max, dec_max]], 0)[0]
					xmin, ymin, xmax, ymax = ellipse_to_box(phi, major, minor, x, y)
					center = patches.Circle((x, y), radius=1, edgecolor='g', facecolor='none')
					#xmin, xmax = xmin -i, xmax -i
					#ymin, ymax = ymin -j, ymax -j 
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
			#print("I: ",i," ",i+IMG_SIZE)
			#print("J: ",j," ",j+IMG_SIZE)
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

def make_fits_2D(hdu):
	hdu.header['NAXIS'] = 2
	hdu.header['WCSAXES'] = 2
	to_delete = ['NAXIS3', 'NAXIS4', 'CRPIX3', 
				'CDELT3', 'CRVAL3', 'CTYPE3', 
				'CRPIX4', 'CDELT4', 'CRVAL4', 'CTYPE4']
	for keyword in to_delete:
		del hdu.header[keyword]
	return hdu

divideImages()
#array = np.load("img_array.npy")
#for i in range(0,10):
	# Save the generated images
	#np.save("img_array.npy",img_array)
#	plt.imshow(array[i], cmap='gist_heat')
#	plt.show()