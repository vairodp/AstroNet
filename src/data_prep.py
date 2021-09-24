#Utilities
import pandas as pd
import numpy as np

#Astropy
from astropy.io import fits
import astropy.wcs as pywcs
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.io.fits.verify import VerifyWarning
from configs.train_config import IMG_SIZE

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
SIGMA = 3.5

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
	ax.imshow(img_array, cmap='gist_heat')
	ax.text(xmin,ymin,text,c=color)

def clip(data, lim):
    data[data < lim] = 0.0
    return data 

def divide_images(img_path, training_set_path, cutouts_path, plot=False):

	#Prep Image
	fits_img = fits.open(img_path)
	fits_img = make_fits_2D(fits_img[0])
	
	#Prep Training Set
	TrainingSet = pd.read_csv(training_set_path, skiprows=17, delimiter='\s+')
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
	RANGE_MIN, RANGE_MAX = get_boundaries(TrainingSet['x'], TrainingSet['y'])
	print(RANGE_MIN, RANGE_MAX)

	fits_img = fits_img.data[0,0]
	
	training_data = {
		'class': [],
		'img_path': [],
		'xmax': [],
		'xmin': [],
		'ymax': [],
		'ymin': []
	}

	TrainingSet = TrainingSet[TrainingSet['SELECTION'] == 1]
		
	fits_img[np.isnan(fits_img)] = 0
	
	
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
				filename = f'{prefix}img-{i}-{j}.fits'
				
				# Remove background noise
				_, _, std = sigma_clipped_stats(img_array, sigma=SIGMA)
				img_array = clip(img_array, lim=std*SIGMA)

				hdu = fits.PrimaryHDU(data=img_array, header=img_fits.wcs.to_header())
				
				hdu.writeto(cutouts_path + filename, overwrite=True)
				if plot:
					_, ax = plt.subplots()
				for _, row in small_ts.iterrows():

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


					training_data['img_path'].append(filename)
					training_data['class'].append(row.CLASS - 1)
					training_data['xmax'].append(xmax / IMG_SIZE)
					training_data['xmin'].append(xmin / IMG_SIZE)
					training_data['ymax'].append(ymax / IMG_SIZE)
					training_data['ymin'].append(ymin / IMG_SIZE)
					
					if plot:
						img = fits.open(cutouts_path + filename)
						img = img[0].data
						plot_images_and_bbox(ax, img, row.CLASS, centroid_x, 
											centroid_y, xmin, ymin, xmax, ymax)
			if plot:
				plt.show()
			
	training_df = pd.DataFrame.from_dict(training_data)
	training_df.to_csv(cutouts_path + 'galaxies.csv', index=False)
	
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