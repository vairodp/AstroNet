#Utilities
import pandas as pd
import numpy as np
import math

#Astropy
from astropy.io import fits
import astropy.wcs as pywcs
from astropy.nddata.utils import Cutout2D
from astropy.io.fits.verify import VerifyWarning
from configs.train_config import IMG_SIZE

#from photutils.background import Background2D

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
# b1: KAPPA = 2, no power
# b2: KAPPA = 2, power=1.2
KAPPA = 2

PRIMARY_BEAM_B1 = '../data/ancillary/PrimaryBeam_B1.fits'
PRIMARY_BEAM_B2 = 'PrimaryBeam_B2.fits'
PRIMARY_BEAM_B5 = 'PrimaryBeam_B5.fits'

def get_boundaries(x, y):
	min_x, max_x = x.min(), x.max()
	min_y, max_y = y.min(), y.max()
	minimum = min(min_x, min_y)
	maximum = max(max_x, max_y)
	minimum = (int(minimum / IMG_SIZE) + 1) * IMG_SIZE
	maximum = (int(maximum / IMG_SIZE) - 1) * IMG_SIZE
	return minimum, maximum


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
	ax.imshow(img_array, origin='lower')
	ax.text(xmin,ymin,text,c=color)

def correct_primary_beam(training_set, pb_wcs, pb_data, x_pixel_res):
	for _, row in tqdm(training_set.iterrows(), desc='Correcting Primary Beam..'):
		x, y = pb_wcs.wcs_world2pix([[row['RA (centroid)'], row['DEC (centroid)'], 0, 0]], 0)[0][0:2]
		pbv = pb_data[int(y)][int(x)]
		flux = row['FLUX'] * pbv
		area_pixel = ((row['BMAJ'] / 3600 / x_pixel_res) * (row['BMIN'] / 3600 / x_pixel_res)) / 1.1
		with np.errstate(divide='ignore', invalid='ignore'):
			row['FLUX'] = np.nan_to_num(flux / area_pixel, posinf=0.0, neginf=0.0)
	return training_set

#For now filters with the threshold but should works with everything I hope
def newDivideImages(img_path, training_set_path, cutouts_path, plot=False):

	COUNTERS = {
		1.0 : 0,
		2.0 : 0,
		3.0 : 0
	}

	#Prep Image
	fits_img = fits.open(img_path)
	fits_img = make_fits_2D(fits_img[0])

	# Open primary beam correction file
	# pb_wcs, pb_data = _setup_pb(beam_correction_file)
	
	#Prep Training Set
	TrainingSet = pd.read_csv(training_set_path, skiprows=17, delimiter='\s+')
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
	RANGE_MIN, RANGE_MAX = get_boundaries(TrainingSet['x'], TrainingSet['y'])
	print(RANGE_MIN, RANGE_MAX)

	fits_img = fits_img.data[0,0]
	#sigma = np.std(fits_img)
	#rms = np.sqrt(np.mean(fits_img ** 2))

	training_data = {
		'class': [],
		'img_path': [],
		'xmax': [],
		'xmin': [],
		'ymax': [],
		'ymin': []
	}

	TrainingSet = TrainingSet[TrainingSet['SELECTION'] == 1]

	global_count = len(TrainingSet)
	filtered_count = 0

	#TrainingSet = correct_primary_beam(TrainingSet, pb_wcs, pb_data, X_PIXEL_RES)

	#sigma = np.std(fits_img[RANGE_MIN:RANGE_MAX, RANGE_MIN:RANGE_MAX])
	
	fits_img[np.isnan(fits_img)] = 0
	#rms = np.sqrt(np.mean(fits_img[RANGE_MIN:RANGE_MAX, RANGE_MIN:RANGE_MAX] ** 2))
	#rms = np.mean(fits_img[RANGE_MIN:RANGE_MAX, RANGE_MIN:RANGE_MAX])

	for i in tqdm(range(RANGE_MIN, RANGE_MAX, IMG_SIZE), desc='Preparing images...'):
		for j in range(RANGE_MIN, RANGE_MAX, IMG_SIZE):
			#Query
			pos = (i+(IMG_SIZE/2), j+(IMG_SIZE/2))
			img_fits = Cutout2D(fits_img, position=pos,size=IMG_SIZE, wcs=WORLD_REF, copy=True)
			img_array = img_fits.data
			small_ts = TrainingSet.query('x < @i+@IMG_SIZE and x >= @i and y < @j+@IMG_SIZE and y >= @j')
			
			if len(small_ts) > 0:
				prefix_index = img_path.find('B')
				prefix = img_path[prefix_index:prefix_index+2]
				filename = f'{prefix}img-{i}-{j}.png'
				#if 'B2' in img_path:
				#	img_array = power(img_array, power_index=3.0, scale_min=0.0)
				plt.imsave(cutouts_path + filename, img_array, origin='lower')
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
					xmax = min(xmax, IMG_SIZE-1)
					ymax = min(ymax, IMG_SIZE-1)

					COUNTERS[row.CLASS] += 1

					training_data['img_path'].append(filename)
					training_data['class'].append(row.CLASS - 1)
					training_data['xmax'].append(xmax / IMG_SIZE)
					training_data['xmin'].append(xmin / IMG_SIZE)
					training_data['ymax'].append(ymax / IMG_SIZE)
					training_data['ymin'].append(ymin / IMG_SIZE)
					
					if plot:
						plot_images_and_bbox(ax, img_array, row.CLASS, centroid_x, 
											centroid_y, xmin, ymin, xmax, ymax)
				
			if plot:
				plt.show()
	training_df = pd.DataFrame.from_dict(training_data)
	training_df.to_csv(cutouts_path + 'galaxies.csv', index=False)
	print("GLOBAL = ", global_count)
	print("FILTERED = ", filtered_count)
	print("COUNTERS: ", COUNTERS)

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

def power(inputArray, power_index=3.0, scale_min=None, scale_max=None):

	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()
	factor = 1.0 / math.pow((scale_max - scale_min), power_index)
	indices0 = np.where(imageData < scale_min)
	indices1 = np.where((imageData >= scale_min) & (imageData <= scale_max))
	indices2 = np.where(imageData > scale_max)
	imageData[indices0] = 0.0
	imageData[indices2] = 1.0
	imageData[indices1] = np.power((imageData[indices1] - scale_min), power_index)*factor

	return imageData

#newDivideImages(img_path='../data/raw/SKAMid_B1_1000h_v3.fits', 
#				training_set_path='../data/raw/TrainingSet_B1_v2.txt',
#				beam_correction_file=PRIMARY_BEAM_B1,
#				cutouts_path='../data/training/B5_1000h/')

#train_test_split(filepath=CUTOUTS_PATH + 'galaxies_560Hz.csv')