#Utilities
import pandas as pd
import numpy as np

#Astropy
from astropy.io import fits
import astropy.wcs as pywcs
from astropy.nddata.utils import Cutout2D
from astropy.io.fits.verify import VerifyWarning

#Suppressing Warnings
import warnings 
warnings.simplefilter('ignore', category=VerifyWarning)

#Plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

#Loading Bar
from tqdm import tqdm

#Globals Values
IMG_SIZE = 128
RANGE_MIN= 16384
RANGE_MAX = 20096
KAPPA = 0.5

#Global Paths
IMG_PATH = './data/raw/SKAMid_B1_1000h_v3.fits'
TRAINING_SET_PATH = './training_set/TrainingSet_B1_v2.txt'
CUTOUTS_PATH = './data/training/B1_1000h/'


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

def adjustSemiAxes(major,minor,category,size,BMAJ_RATIO,BMIN_RATIO,G_SQUARED):

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

def plot_images_and_bbox(ax, img_array, category, centroid_x, centroid_y, xmin, ymin, xmax, ymax):

	center = patches.Circle((centroid_x,centroid_y),radius=1, edgecolor='g',facecolor = "none")
	
	if category == 1:
		color = 'yellow'
		text = '1: SS-AGN'
	elif category == 2:
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
	ax.add_patch(center)
	ax.add_patch(box)
	ax.imshow(img_array,cmap='gist_heat', origin='lower')
	ax.text(xmin,ymin,text,c=color)

#For now filters with the threshold but should works with everything I hope
def newDivideImages(plot=False):

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
	#TrainingSet = TrainingSet.drop_duplicates(subset=['x', 'y'])

	#Constants
	X_PIXEL_RES = abs(fits_img.header['CDELT1'])
	BMAJ_TOT = abs(fits_img.header['BMAJ'])
	BMIN_TOT = abs(fits_img.header['BMIN'])
	BMAJ_RATIO = BMAJ_TOT / X_PIXEL_RES
	BMIN_RATIO = BMIN_TOT / X_PIXEL_RES
	G_SQARED = (2 * X_PIXEL_RES * 3600) ** 2
	WORLD_REF = pywcs.WCS(fits_img.header).deepcopy()

	fits_img = fits_img.data[0,0]
	#sigma = np.std(fits_img)
	#rms = np.sqrt(np.mean(fits_img ** 2))

	training_data = {
		'class': [],
		'img_path': [],
		'height': [],
		'width': [],
		'xmax': [],
		'xmin': [],
		'ymax': [],
		'ymin': []
	}

	global_count = len(TrainingSet[TrainingSet['SELECTION'] == 1])
	filtered_count = 0

	#sigma = np.std(fits_img[RANGE_MIN:RANGE_MAX, RANGE_MIN:RANGE_MAX])
	rms = np.sqrt(np.mean(fits_img[RANGE_MIN:RANGE_MAX, RANGE_MIN:RANGE_MAX] ** 2))

	for i in tqdm(range(RANGE_MIN,RANGE_MAX,IMG_SIZE), desc='Loading...'):
		for j in range(RANGE_MIN,RANGE_MAX,IMG_SIZE):
			#Query
			pos = (i+(IMG_SIZE/2), j+(IMG_SIZE/2))
			img_fits = Cutout2D(fits_img, position=pos,size=IMG_SIZE, wcs=WORLD_REF, copy=True)
			img_array = img_fits.data
			small_ts = TrainingSet.query('x < @i+@IMG_SIZE and x >= @i and y < @j+@IMG_SIZE and y >= @j and FLUX > (@rms * @KAPPA) and SELECTION == 1')
			
			filename = f'img-{i}-{j}.png'
			plt.imsave(CUTOUTS_PATH + filename, img_array)
			if plot:
				_, ax = plt.subplots()
			for _, row in small_ts.iterrows():

				filtered_count += 1

				centroid_x = int(row['x']-i)
				centroid_y = int(row['y']-j)
				
				#Box
				major = (row['BMAJ'] / 3600 / X_PIXEL_RES) / 2
				minor = (row['BMIN'] / 3600 / X_PIXEL_RES) / 2
				phi = np.radians(row['PA'])

				major,minor = adjustSemiAxes(major,minor,row.CLASS,row.SIZE, BMAJ_RATIO, BMIN_RATIO, G_SQARED)

				#Crop Box around the corner if oversized 
				xmin, ymin, xmax, ymax = ellipse_to_box(phi, major, minor, centroid_x, centroid_y)
				xmin = max(xmin, 0)
				ymin = max(ymin, 0)
				xmax = min(xmax,IMG_SIZE-1)
				ymax = min(ymax,IMG_SIZE-1)

				training_data['img_path'].append(filename)
				training_data['class'].append(row.CLASS)
				training_data['height'].append(ymax-ymin)
				training_data['width'].append(xmax - xmin)
				training_data['xmax'].append(xmax)
				training_data['xmin'].append(xmin)
				training_data['ymax'].append(ymax)
				training_data['ymin'].append(ymin)
				
				if plot:
					plot_images_and_bbox(ax, img_array, row.CLASS, centroid_x, 
										centroid_y, xmin, ymin, xmax, ymax)
				
			if plot:
				plt.show()
	training_df = pd.DataFrame.from_dict(training_data)
	training_df.to_csv(CUTOUTS_PATH + 'galaxies_560Hz.csv')
	print("GLOBAL = ", global_count)
	print("FILTERED = ", filtered_count)

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
				'CRPIX4', 'CDELT4', 'CRVAL4',
				'CTYPE4']
	for keyword in to_delete:
		del hdu.header[keyword]
	return hdu

def train_test_split(filepath, test_size=0.20):
	data = pd.read_csv(filepath)
	print(len(data.img_path))
	file_paths = data.img_path.unique()
	sample_num = int(len(file_paths) * test_size)
	test_paths = np.random.choice(file_paths, size=sample_num)
	print(len(test_paths))
	print(len(file_paths))

newDivideImages()
#train_test_split(filepath=CUTOUTS_PATH + 'galaxies_560Hz.csv')