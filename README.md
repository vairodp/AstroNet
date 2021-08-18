## Requirements

The code requires python >= 2.7 as well as the following python libraries:

* astropy
* imgaug
* matplotlib
* numpy
* pandas
* scikit-learn
* tensorflow
* tensorflow-datasets==4.3.0
* tqdm
* opencv-python


**Install Modules:** 

`pip install -U pip`

`pip install -r requirements.txt`


# SKA-DC1
Deep Learning Approach to the Square Kilometer Array Data Challenge #1 

| Column | ID             | unit of measure | Description                                                                                     |
| :----: | :------------- | :-------------: | :---------------------------------------------------------------------------------------------- |
|   1    | ID             |      none       | Source ID                                                                                       |
|   2    | RA (core)      |      degs       | Right ascension of the source core                                                              |
|   3    | DEC (core)     |      degs       | DEcination of the source core                                                                   |
|   4    | RA (centroid)  |      degs       | Right ascension of the source centroid                                                          |
|   5    | DEC (centroid) |      degs       | Declination of the source centroid                                                              |
|   6    | FLUX           |       Jy        | integrated flux density                                                                         |
|   7    | Core frac      |      none       | integrated flux density of core/total                                                           |
|   8    | BMAJ           |     arcsec      | major axis dimension                                                                            |
|   9    | BMIN           |     arcsec      | minor axis dimension                                                                            |
|   10   | PA             |      degs       | PA (measured clockwise from the longitude-wise direction)                                       |
|   11   | SIZE           |      none       | 1,2,3 for LAS, Gaussian, Exponential                                                            |
|   12   | CLASS          |      none       | 1,2,3 for SS-AGNs, FS-AGNs,SFGs                                                                 |
|   13   | SELECTION      |      none       | 0,1 to record that the source has not/has been injected in the simulated map due to noise level |
|   14   | x              |      none       | pixel x coordinate of the centroid, starting from 0                                             |
|   15   | y              |      none       | pixel y coordinate of the centroid,starting from 0                                              |
