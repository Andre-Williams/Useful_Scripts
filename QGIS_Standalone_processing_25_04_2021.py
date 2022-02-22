
######### INITIALISE QGIS STANDALONE ################

import sys
from qgis.core import (
     QgsApplication, 
     QgsProcessingFeedback, 
     QgsVectorLayer,
     QgsField,
     QgsFields,
     QgsProperty,
     QgsProcessing,
     QgsProcessingFeatureSourceDefinition,
     QgsProcessingOutputLayerDefinition
)

#start QGIS instance without GUI
#Make sure the prefix is correct. Even though QGIS is in '/usr/share/qgis',
#the prefix needs to be '/usr' (assumes Debian OS)

QgsApplication.setPrefixPath('/usr', True)
myqgs = QgsApplication([], False)
myqgs.initQgis()

######### INITIALISE THE PROCESSING FRAMEWORK ################

# Append the path where processing plugin can be found (assumes Debian OS)
sys.path.append('/usr/share/qgis/python/plugins')

#import modules needed
import processing
from processing.core.Processing import Processing
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj
import numpy as np

#start the processing module
processing.core.Processing.Processing.initialize()

################# Creating a Dictionary for header names ################################
def Convert(lst): #Converts list to dictionary
	global field_dct
	field_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
	return field_dct

df = pd.read_csv('Sample_points.csv')
field_headings = df.columns.tolist()

fields_index_list = []
count = 0

for field in field_headings:
	fields_index_list.append(count)
	fields_index_list.append(field)
	count += 1

Convert(fields_index_list)

print(field_dct)
#Field Names Dictionary
'''
{'Sample_no': 0, 'x_coord': 1, 'y_coord': 2, 'x': 3, 'y': 4,
'Total_Exchange_Capacity': 5, 'OrgMat_per': 6, 'pH': 7,
'Sulfur_ppm': 8, 'P_ppm': 9, 'Ca_kg/ha': 10, 'Ca_ppm': 11,
'Mg_kg/ha': 12, 'Mg_ppm': 13, 'K_kg/ha': 14, 'K_ppm': 15,
'Na_kg/ha': 16, 'Na_ppm': 17, 'Ca_per': 18, 'Mg_per': 19,
'K_per': 20, 'Na_per': 21, 'B_ppm': 22, 'Fe_ppm': 23,
'Mn_ppm': 24, 'Cu_ppm': 25, 'Zn_ppm': 26}
'''

################# CSV TO SHAPEFILE + WGS84 CRS TO UTM33S CRS ############################

#Convert Sample Points WGS84 co-ordinates to UTMS33 co-ordinates
myProj = Proj("+proj=utm +zone=33S, +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
UTMx, UTMy = myProj(np.array(df['x']), np.array(df['y']))

y_coords = []

for y in UTMy:
	val = y + 10_000_000
	y_coords.append(val)

UTMy_2 = np.array(y_coords)

df['UTMx'], df['UTMy'] = UTMx, UTMy_2 

gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip (df.UTMx, df.UTMy)])
gdf.set_crs(epsg=32733, inplace=True)

#Converting GeoDataFrame to a ESRI Shapefile
gdf.to_file('Sample_points.shp',driver='ESRI Shapefile')

############################### QGIS Processing ##################################

#Fields_that are to be interpolated
heading_index = len(field_headings)

for seq in range(5,heading_index+1):

	IDW_input = 'Sample_points.shp::~::0::~::{}::~::0'.format(seq)
	IDW_output = 'interpolated_raster_{}'.format(field_dct[seq])

	Farm_boundary = 'Farm_boundary.shp'

	IDW = processing.run("qgis:idwinterpolation",\
	{'INTERPOLATION_DATA': IDW_input,\
	'DISTANCE_COEFFICIENT':2.5,\
	'EXTENT':'323266.347700000, 323861.727600000,8335174.220800000,8335527.756100000 [EPSG:32733]',\
	'PIXEL_SIZE':22.43776,\
	'OUTPUT': IDW_output}) #QgsProcessing.TEMPORARY_OUTPUT

