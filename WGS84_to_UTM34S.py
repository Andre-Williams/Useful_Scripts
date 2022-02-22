#WGS84 to UTM34S

import pandas as pd
import numpy as np
from datetime import datetime
import rasterio
from pyproj import Proj
import sys

def WGS84_2_UTM34S():
	df = pd.read_csv('Markers_GPS_points.csv')
	print(df)

	#Convert GPS co-ordinate readings CRS from WGS84 to UTM34S
	myProj = Proj("+proj=utm +zone=34S, +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
	UTMx, UTMy = myProj(np.array(df['wgs84_long']), np.array(df['wgs84_lat']))

	UTMx = list(UTMx)

	UTMy_2 = []

	for y in UTMy:
		val = y + 10000000
		UTMy_2.append(val)

	UTMy = UTMy_2


	df['UTMx'] = UTMx
	df['UTMy'] = UTMy
	print(UTMx, UTMy)

	df.to_csv('UTM34S_points.csv', index=False)

WGS84_2_UTM34S()
