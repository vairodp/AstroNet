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
IMG_SIZE = 128
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

def adjustSemiAxis(major,minor,category,size,BMAJ_RATIO,BMIN_RATIO,G_SQUARED):

	if category == 1 and size == 1:
		bmaj, bmin = major/2.0,minor/2.0
	elif category == 2 and size == 2:
		bmaj, bmin = major/2.0,major/2.0
	elif category == 3 and size == 3:
		bmaj, bmin = major*(2**0.5),minor*(2**0.5)
	else:
		bmaj, bmin =  major,minor
	
	major = np.sqrt(bmaj ** 2 + G_SQUARED) * BMAJ_RATIO
	minor = np.sqrt(bmin ** 2 + G_SQUARED) * BMIN_RATIO

	return major, minor

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

	#Constants
	X_PIXEL_RES = abs(fits_img.header['CDELT1'])
	BMAJ_TOT = abs(fits_img.header['BMAJ'])
	BMIN_TOT = abs(fits_img.header['BMIN'])
	BMAJ_RATIO = BMAJ_TOT / X_PIXEL_RES
	BMIN_RATIO = BMIN_TOT / X_PIXEL_RES
	G_SQARED = (2 * X_PIXEL_RES * 3600) ** 2
	WORLD_REF = pywcs.WCS(fits_img.header).deepcopy()

	fits_img = fits_img.data[0,0]

	for i in range(16300,32768,IMG_SIZE):
		for j in range(16300,32768,IMG_SIZE):
			#Query
			small_ts = TrainingSet.query('x < @i+@IMG_SIZE and x >= @i and y < @j+@IMG_SIZE and y >= @j and FLUX > @FLUX_TRESHOLD and SELECTION == 1')
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
						flux = row['FLUX']
						print('x: ', str(row['x']), "y: ", str(row['y']), " Flux =", flux)
						print("Normal i and j :",i, "  ", j)
						print(str(i+IMG_SIZE), "  ", str(j+IMG_SIZE))
						print("----------------------------")
						#Center
						centroid_x = int(row['x']-i)
						centroid_y = int(row['y']-j)
						center = patches.Circle((centroid_x,centroid_y),radius=1, edgecolor='g',facecolor = "none")

						#Box
						major = (row['BMAJ'] / 3600 / X_PIXEL_RES) / 2
						minor = (row['BMIN'] / 3600 / X_PIXEL_RES) / 2
						phi = np.radians(row['PA'])

						major,minor = adjustSemiAxis(major,minor,row.CLASS,row.SIZE, BMAJ_RATIO, BMIN_RATIO, G_SQARED)

						#Fixed 
						xmin, ymin, xmax, ymax = ellipse_to_box(phi, major, minor, centroid_x, centroid_y)
						xmin = max(xmin, 0)
						ymin = max(ymin, 0)
						xmax = min(xmax,IMG_SIZE-1)
						ymax = min(ymax,IMG_SIZE-1)

						if row.CLASS == 1:
							color = 'yellow'
							text = '1: SS-AGN'
						elif row.CLASS == 2:
							color = 'green'
							text = '2: FS-AGN'
						else:
							color = 'blue'
							text = '3: SFG'
						box = patches.Rectangle((xmin, ymin),
                                                    width=xmax - xmin,
                                                    height=ymax - ymin,
                                                    linewidth=1,
                                                    edgecolor=color,
                                                    facecolor='none')
						#Update plot
						ax.add_patch(center)
						ax.add_patch(box)
						ax.imshow(img_array,cmap='gist_heat', origin='lower')
						ax.text(xmin,ymin,text,c=color)
						ax.text(xmax, ymax, 'XMAX', c=color)
						
						#Debugging
						#deb_plot(row['x'],row['y'])
				plt.show()
	return
			
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