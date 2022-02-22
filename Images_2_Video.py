import glob
import cv2

#Creating Video from Images 


img_arrayR = []

for i in range(0,20):   
    t = str(i)
    #img = cv2.imread('Calibration_Images/Right/chessboard-R' + t + '.png')
    img = cv2.imread('Rectified_Images/Right/calibresult-R-' + t + '.png')
    
    height, width, layers = img.shape
    size = (width,height)
    img_arrayR.append(img)
 
 
#outR = cv2.VideoWriter('R_Vid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
outR = cv2.VideoWriter('R_Vid2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size) 
 
for i in range(len(img_arrayR)):
    outR.write(img_arrayR[i])
outR.release()

img_arrayL = []
for i in range(0,20):   
    t = str(i)
    #img = cv2.imread('Calibration_Images/Left/chessboard-L' + t + '.png')
    img = cv2.imread('Rectified_Images/Left/calibresult-L-' + t + '.png')
    
    height, width, layers = img.shape
    size = (width,height)
    img_arrayL.append(img)
 
 
#outL = cv2.VideoWriter('L_Vid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
outL = cv2.VideoWriter('L_Vid2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
 
for i in range(len(img_arrayL)):
    outL.write(img_arrayL[i])
outL.release()
