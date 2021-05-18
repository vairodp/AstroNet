#Utilities
import pandas as pd
import numpy as np

#Astropy
from astropy.io import fits
import astropy.wcs as pywcs
from astropy.nddata.utils import Cutout2D
import aplpy

#Plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

#Globals
IMG_SIZE = 200
IMG_PATH = './data/SKAMid_B1_1000h_v3.fits'
TRAINING_SET_PATH = './training_set/TrainingSet_B1_v2.txt'

#Tweak this
FLUX_TRESHOLD = 0.001

def prepareImageFirstMethod():
	img = aplpy.FITSFigure(IMG_PATH, downsample=25)
	img.show_colorscale(cmap='gist_heat')
	img.save('./data/imageResult1.png')

def showImage():
	img = mpimg.imread('/data/imageResult1.png')
	imgplot = plt.imshow(img)
	plt.show()

def showDataSet():
	TrainingSet=pd.read_csv(TRAINING_SET_PATH, skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']

	print(TrainingSet)
	print(TrainingSet.len())


#FOR DEBUGGING: given x y pixel coordinates, plot a cutout with x y center
def deb_plot(x, y):
	siz = IMG_SIZE
	fits_img = fits.open(IMG_PATH)
	fits_img = make_fits_2D(fits_img[0])

	X_PIXEL_RES = abs(fits_img.header['CDELT1'])
	WORLD_REF = pywcs.WCS(fits_img.header).deepcopy()

	fits_img = fits_img.data[0, 0]

	pos = (x, y)
	img_fits = Cutout2D(fits_img, position=pos, size=siz,
	                    wcs=WORLD_REF, copy=True)
	
	img_array = img_fits.data

	_, ax = plt.subplots()
	ax.imshow(img_array, cmap='gist_heat')

	plt.show()

	return

#For now filters with the threshold but should works with everything I hope
def newDivideImages():

	#Prep Image
	fits_img = fits.open(IMG_PATH)
	fits_img = make_fits_2D(fits_img[0])
	
	#Prep Training Set
	TrainingSet = pd.read_csv(TRAINING_SET_PATH, skiprows=17, delimiter='\s+')
	TrainingSet = TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns = ['ID', 'RA (core)', 'DEC (core)', 'RA (centroid)', 'DEC (centroid)',
                        'FLUX', 'Core frac', 'BMAJ', 'BMIN', 'PA', 'SIZE', 'CLASS', 'SELECTION', 'x', 'y']
	TrainingSet['x'] = TrainingSet['x'].astype(int)
	TrainingSet['y'] = TrainingSet['y'].astype(int)
	TrainingSet['FLUX'] = TrainingSet['FLUX'].astype(float)

	X_PIXEL_RES = abs(fits_img.header['CDELT1'])
	WORLD_REF = pywcs.WCS(fits_img.header).deepcopy()

	fits_img = fits_img.data[0,0]

	for i in range(16300,32768,IMG_SIZE):
		for j in range(16300,32768,IMG_SIZE):
			#Query
			small_ts = TrainingSet.query('x < @i+@IMG_SIZE and x >= @i and y < @j+@IMG_SIZE and y >= @j and FLUX > @FLUX_TRESHOLD')
			#Not-empty section
			if (len(small_ts)) > 0:
				pos = (i+(IMG_SIZE/2), j+(IMG_SIZE/2))
				img_fits = Cutout2D(fits_img, position=pos,size=IMG_SIZE, wcs=WORLD_REF, copy=True)
				img_array = img_fits.data
				_, ax = plt.subplots()
				for _, row in small_ts.iterrows():
					#Found something relevant
					if row['FLUX'] > FLUX_TRESHOLD:

						#Print Debugging
						#flux = row['FLUX']
						#print('x: ', str(row['x']), "y: ", str(row['y']), " Flux =", flux)
						#print("Normal i and j :",i, "  ", j)
						#print(str(i+IMG_SIZE), "  ", str(j+IMG_SIZE))
						
						#Center
						centroid_x = int(row['x']-i)
						centroid_y = int(row['y']-j)
						center = patches.Circle((centroid_x,centroid_y),radius=1, edgecolor='g',facecolor = "none")

						#Box
						major = (row['BMAJ'] / 3600 / X_PIXEL_RES) / 2
						minor = (row['BMIN'] / 3600 / X_PIXEL_RES) / 2
						phi = np.radians(row['PA'])
						xmin, ymin, xmax, ymax = ellipse_to_box(phi, major, minor, centroid_x, centroid_y)
						box = patches.Rectangle((xmin, ymin),
                                                    width=xmax - xmin,
                                                    height=ymax - ymin,
                                                    linewidth=1,
                                                    edgecolor='y',
                                                    facecolor='none')
						#Update plot
						ax.add_patch(center)
						ax.add_patch(box)
						ax.imshow(img_array,cmap='gist_heat')
						
						#Debugging
						#deb_plot(row['x'],row['y'])
				plt.show()
	return

def divideImages():
	# Divide the fits image in IMG_SIZE X IMG_SIZE images
	fits_img = fits.open(IMG_PATH)
	fits_img = make_fits_2D(fits_img[0])
	TrainingSet=pd.read_csv(TRAINING_SET_PATH,skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']
	TrainingSet['x'] = TrainingSet['x'].astype(int)
	TrainingSet['y'] = TrainingSet['y'].astype(int)

	X_PIXEL_RES = abs(fits_img.header['CDELT1'])
	WORLD_REF = pywcs.WCS(fits_img.header).deepcopy()

	fits_img=fits_img.data[0,0]

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
					x = row['x'] % IMG_SIZE
					y = row['y'] % IMG_SIZE
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

newDivideImages()