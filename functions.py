import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import math
#import os

def calibrate(workingDir):
    '''
    This function uses the the .mp4 calibration video created by combining all the calibration images
    captured in the field.
    '''
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((4 * 3, 3), np.float32)
    objp[:, :2] = np.mgrid[0:4, 0:3].T.reshape(-1, 2)

    objp = objp * 0.203  # 0.065 length side of square on smaller old chessboard in meters.
                         # 0.203 length side of square on bigger new chessboard in meters.

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpointsL = [] # 2d points in image plane
    imgpointsR = []

    # Start calibration from the camera
    print("Starting calibration for the 2 cameras... ")

    CamL= cv2.VideoCapture(workingDir + '/dataset_lft/calibration_data/lft_camera_calibration.mp4')
    CamR= cv2.VideoCapture(workingDir + '/dataset_rht/calibration_data/rht_camera_calibration.mp4')

    i = 1

    while (CamL.isOpened()):

        retL, L_path = CamL.read()
        retR, R_path = CamR.read()

        if (retL == True) & (retL == True):
            ChessImaR= cv2.cvtColor(R_path,cv2.COLOR_BGR2GRAY)
            ChessImaL= cv2.cvtColor(L_path,cv2.COLOR_BGR2GRAY)

            # Define the number of chessboard corners we are looking for
            retR, cornersR = cv2.findChessboardCorners(ChessImaR, (4, 3), None)
            retL, cornersL = cv2.findChessboardCorners(ChessImaL, (4, 3), None)

            if (True == retR) & (True == retL):  # Determines if chessboard was found

                print("Chessboard found for image pair:", i)
                i += 1
                objpoints.append(objp)
                # Increases accuracy of fond corners on chess board
                cornersR_2 = cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
                cornersL_2 = cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
                imgpointsR.append(cornersR_2)
                imgpointsL.append(cornersL_2)

                # # To view the found chessboard corners
                # cv2.drawChessboardCorners(ChessImaR, (5, 3), cornersR_2, retR)
                # cv2.drawChessboardCorners(ChessImaL, (5, 3), cornersL_2, retL)
                # cv2.imshow("ChessImaR", ChessImaR)
                # cv2.imshow("ChessImaL", ChessImaL)
                # cv2.waitKey(0)
            else:
                print("Chessboard NOT found for image pair: ", i)
                i += 1

        else:
            break

    CamR.release()
    CamL.release()
    # cv2.destroyAllWindows()

# ***********************************************************************
# ***** Determine the new values for different intrinsic parameters *****
# ***********************************************************************

    #****** Right Side ******

    # returns the camera matrix, distortion coefficients, rotation and translation vectors
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpointsR, ChessImaR.shape[::-1], None, None
    )
    hR, wR = ChessImaR.shape[:2]

    # used to refine/optimise the camera matrix based on a free scaling parameter.
    # ROI can be used to crop image.
    OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    #****** Left Side *******

    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpointsL, ChessImaL.shape[::-1], None, None
    )
    hL, wL = ChessImaL.shape[:2]

    OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    # Find error
    def get_proj_error(objpoints, rvecs, tvecs, mtx, dist, imgpoints):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        return mean_error / len(objpoints)


    errorR = get_proj_error(objpoints, rvecsR, tvecsR, mtxR, distR, imgpointsR)
    errorL = get_proj_error(objpoints, rvecsL, tvecsL, mtxL, distL, imgpointsL)

    print("Projection error R {}".format(errorR))
    print("Projection error L {}".format(errorL))


# ********************************************
# ***** Calibrate the Cameras for Stereo *****
# ********************************************

    # StereoCalibrate function ######### https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5  <--- LINK FOR DESCRIPTIONS OF PARAMETERS AND OUTPUTS
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5
    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(
        objpoints,  # This finds the intrinsic parameters for each of the two cameras and the extrinsic parameters between the two cameras.
        imgpointsL,
        imgpointsR,
        mtxL,
        distL,
        mtxR,
        distR,
        ChessImaR.shape[::-1],
        criteria_stereo,
        flags,
    )

    # StereoRectify function
    rectify_scale = 0  # if 0 image croped, if 1 image not croped
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
        MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale, (0, 0)
    )  # last paramater is alpha, if 0= croped, if 1= not croped


#********************************************************
#****** Saving Intrinsic and Extrinsic parameters *******
#********************************************************

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    filename = workingDir + "/Intrinsic_&_Extrinsic_parameters" + "_{}".format(current_time)
    np.savez(
        filename,
        MLS,
        dLS,
        RL,
        PL,
        MRS,
        dRS,
        RR,
        PR
    )

    print("Cameras Ready to use")


def read_in_ex_param(in_ex_file):
    '''
    Reads in the intrinsic and extrinsic parameters from the .npz file and assigns each array
    to their corresponding variable as seen below. The .npz file is the output of the 'calibration' function.
    '''

    global MLS, dLS, RL, PL, MRS, dRS, RR, PR
    npzfile = np.load(in_ex_file)

    MLS = npzfile['arr_0']
    dLS = npzfile['arr_1']
    RL = npzfile['arr_2']
    PL = npzfile['arr_3']
    MRS = npzfile['arr_4']
    dRS = npzfile['arr_5']
    RR = npzfile['arr_6']
    PR = npzfile['arr_7']

    print(MLS, dLS, RL, PL, MRS, dRS, RR, PR)

def rectification(mode,lft_vid,rht_vid,in_ex_file):
    '''
    Rectifies the aircraft approach videos or images using the intrinsic and extrinsic parameters.
        - Mode 1: Rectifies the aircraft approach video pairs.
        - Mode 2: Rectifies the marker image pairs.

    '''

    read_in_ex_param(in_ex_file)

    if mode == 1:
        print('Rectifying aircraft apprach video pair.')
        CamL= cv2.VideoCapture(lft_vid)
        CamR= cv2.VideoCapture(rht_vid)   # Wenn 0 then Right Cam and wenn 2 Left Cam

        frame_width = int(CamR.get(3))
        frame_height = int(CamR.get(4))
        frame_size = (frame_width,frame_height)
        fps = 15

        outputL = cv2.VideoWriter('stereo-vision-tracker/input/lft_rect.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
        outputR = cv2.VideoWriter('stereo-vision-tracker/input/rht_rect.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

        while (CamL.isOpened() & CamR.isOpened()):

            retL, L_frame = CamL.read()
            retR, R_frame = CamR.read()

            if (retL == True) & (retR == True):

                # initUndistortRectifyMap function
                Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, frame_size, cv2.CV_16SC2)
                Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, frame_size, cv2.CV_16SC2)
                #
                # Rectify the images on rotation and alignement
                dstL= cv2.remap(L_frame,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the calibration parameters founds during the initialisation
                dstR= cv2.remap(R_frame,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

                # for line in range(0, int(dstR.shape[0]/50)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
                #     dstL[line*50,:]= (0,0,250)
                #     dstR[line*50,:]= (0,0,250)
                #     cv2.imwrite('Rectified_images/Redlines_1.5m/Pair-' + str(i) + '_rect' + '.png', np.hstack([dstL, dstR]))

                outputL.write(dstL)
                outputR.write(dstR)

            else:
                break


        CamR.release()
        CamL.release()
        print('Rectification Complete')

    elif mode == 2:
        print('Rectifying runway markers image pair.')
        CamL= cv2.VideoCapture(lft_vid)
        CamR= cv2.VideoCapture(rht_vid)   # Wenn 0 then Right Cam and wenn 2 Left Cam

        frame_width = int(CamR.get(3))
        frame_height = int(CamR.get(4))
        frame_size = (frame_width,frame_height)

        while (CamL.isOpened() & CamR.isOpened()):

            retL, L_frame = CamL.read()
            retR, R_frame = CamR.read()

            if (retL == True) & (retR == True):

                # initUndistortRectifyMap function
                Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, frame_size, cv2.CV_16SC2)
                Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, frame_size, cv2.CV_16SC2)
                #
                # Rectify the images on rotation and alignement
                dstL= cv2.remap(L_frame,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the calibration parameters founds during the initialisation
                dstR= cv2.remap(R_frame,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

                # for line in range(0, int(dstR.shape[0]/50)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
                #     dstL[line*50,:]= (0,0,250)
                #     dstR[line*50,:]= (0,0,250)
                #     cv2.imwrite('Rectified_images/Redlines_1.5m/Pair-' + str(i) + '_rect' + '.png', np.hstack([dstL, dstR]))

                cv2.imwrite("stereo-vision-tracker/Rectified_images/Marker_L_rect.png",dstL)
                cv2.imwrite('stereo-vision-tracker/Rectified_images/Marker_R_rect.png',dstR)

            else:
                break


        CamR.release()
        CamL.release()
        print('Rectification Complete')

def coords_mouse_disp_L(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        L_x_px.append(x)
        L_y_px.append(y)
        print('Point left image: ', [x,y])

def coords_mouse_disp_R(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        R_x_px.append(x)
        R_y_px.append(y)
        print('Point right image: ', [x,y])

def marker_selection(L_img, R_img, in_ex_file):
    '''
    Used to extract markers pixel coordinates in left and right rectified image pairs produced in mode 2 of the 'rectification' function.
    These pixel co-ordiantes are then used to calculate depth.

    1) You will first select all the markers in left image by double-clicking on them. Once done selcting all markers in left image
    close the image window.

    2) The right image window will then open up where you will then have to select the same markers in the SAME ORDER as you selected in the left image.
    Once all markers are selcted in right image close the image window.

    3) These marker pixel coordinates are then stored and used to calculate depth (x,y, and z translation values) using the cv2.triangulatePoints fucntion.

    4) A CSV file will then be produced ("Marker_Depth.csv") containing the pixel cordinates of the markers in the left and right image as well as the real-world x, y, and z teanslation
    from the left cameras position.

    N.B. These estimated marker depth values can be used for accuracy assessment as the real-world x and z values can be manually measured infiled
    before data is captured and compared to the estimated depth vaules.
    '''
    global L_x_px, L_y_px, R_x_px, R_y_px

    imgL= cv2.imread(L_img)
    imgR= cv2.imread(R_img)

    L_x_px = []
    L_y_px = []
    R_x_px = []
    R_y_px = []

    R_px_coord = []

    #Opens left img window
    cv2.imshow('Left',imgL)
    cv2.setMouseCallback("Left",coords_mouse_disp_L,imgL)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    #Open right img window
    cv2.imshow('Right',imgR)
    cv2.setMouseCallback("Right",coords_mouse_disp_R,imgR)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    read_in_ex_param(in_ex_file)
    df = pd.DataFrame()

    df['left_x'] = L_x_px
    df['left_y'] = L_y_px
    df['right_x'] = R_x_px
    df['right_y'] = R_y_px

    df_len = len(df)

    real_x = []
    real_y = []
    real_z = []

    for p in range(df_len):
        left_x = float(df['left_x'][p])
        left_y = float(df['left_y'][p])
        right_x = float(df['right_x'][p])
        right_y = float(df['right_y'][p])

        l_coord = [left_x, left_y]
        r_coord = [right_x, right_y]

        tri1 = cv2.triangulatePoints(PL, PR, np.array(l_coord), np.array(r_coord))

        x, y, z = tri1[0][0]/tri1[3][0], tri1[1][0]/tri1[3][0], tri1[2][0]/tri1[3][0]

        HAG = 0.75 #Cameras HEIGHT ABOVE THE GROUND
        real_x.append(x)
        real_y.append((y*-1) - HAG) #To invert the height values so that positive values mean increase in height and negative mean decrease.
        real_z.append(z)

    df['X_rw'] = real_x
    df['Y_rw'] = real_y
    df['Z_rw'] = real_z

    df.to_csv('stereo-vision-tracker/input/Marker_Depth.csv', index=False)


def triangulate_aircraft(aircraft_px_coord_csv, marker_depth_csv, in_ex_file):
    '''
    Using the pixel coordinates of the centre of mass of the aircraft, determined using the ML algorithm, the x, y, and z translation values (depth) are
    calculated using the left camera position as the origin. The translation values are intially calculated from the left camera but but another translation
    is done to add columns containing translation from the landing point (LP) as well as the aircrafts offset/displacement from the ideal flight path.
    '''

    read_in_ex_param(in_ex_file)
    ACdf = pd.read_csv(aircraft_px_coord_csv)

    ACdf_len = len(ACdf)

    real_x = []
    real_y = []
    real_z = []

    for p in range(ACdf_len):
        left_x = float(ACdf['left_x'][p])
        left_y = float(ACdf['left_y'][p])
        right_x = float(ACdf['right_x'][p])
        right_y = float(ACdf['right_y'][p])

        l_coord = [left_x, left_y]
        r_coord = [right_x, right_y]

        tri1 = cv2.triangulatePoints(PL, PR, np.array(l_coord), np.array(r_coord))

        x, y, z = tri1[0][0]/tri1[3][0], tri1[1][0]/tri1[3][0], tri1[2][0]/tri1[3][0]

        real_x.append(x)
        real_y.append(y*-1) #To invert the height values so that positive values mean increas in heigh and negative mean decrease.
        real_z.append(z)

    HAG = 1 # Height of camera Above Ground

    Mdf = pd.read_csv(marker_depth_csv)
    LP_Zdist = Mdf['Z_rw'][0]
    LP_Xdist = Mdf['X_rw'][0]

    ACdf['X_rw'] = real_x
    ACdf['Y_rw'] = real_y
    ACdf['Z_rw'] = real_z

    ACdf['X_LP'] = ACdf['X_rw'] + abs(LP_Xdist)
    ACdf['Y_LP'] = (ACdf['Y_rw']) + HAG #adds the height of the camera above the ground (HAG).
    ACdf['Z_LP'] = ACdf['Z_rw'] - abs(LP_Zdist) #Making the the landing point the origin (0,0) of the CRS

    #Tanslating
    Cdf = pd.read_csv(marker_depth_csv)

    XZpoint1 = [Cdf['X_rw'][0] + abs(Cdf['X_rw'][0]), Cdf['Z_rw'][0] - abs(Cdf['Z_rw'][0])]
    XZpoint2 = [Cdf['X_rw'][1] + abs(Cdf['X_rw'][0]), Cdf['Z_rw'][1] - abs(Cdf['Z_rw'][0])]

    XZm = (XZpoint2[1] - XZpoint1[1]) / (XZpoint2[0] - XZpoint1[0]) #m = gradient

    XZc = XZpoint1[1] - (XZm*XZpoint1[0])   #y-intercept

    #y = XZm*x + XZc

    Euclidean_Z = []

    for row in range(ACdf_len):
        if pd.isnull(ACdf['Z_LP'][row]):
            Z = np.NaN
            Euclidean_Z.append(Z)

        else:
            Z = math.sqrt(ACdf['X_LP'][row]**2 + ACdf['Z_LP'][row]**2)
            Euclidean_Z.append(Z)

    ACdf['Euclidean_Z'] = Euclidean_Z

    #OffsetX calculation i.e. shortest distance from point to  ideal flight path.
    d_list = []
    b = 1
    for row in range(ACdf_len):
        if pd.isnull(ACdf['X_LP'][row]):
            offset = np.NaN
            d_list.append(offset)
        else:
            General_Formula = -XZm * ACdf['X_LP'][row] + ACdf['Z_LP'][row] - XZc
            d = abs(General_Formula / (math.sqrt(XZm * XZm + b * b)))

            #Point is above the line if:
            #a*x_1+b*y_1+c>0 and b>0 For our application b will always be more than 1.
            #a*x_1+b*y_1+c<0 and b<0
            #https://www.emathzone.com/tutorials/geometry/position-of-point-with-respect-to-line.html

            if General_Formula > 0:
                offset = d *-1
                d_list.append(offset)

            elif General_Formula < 0:
                offset = d
                d_list.append(offset)


    ACdf['OffsetX'] = d_list
    idealness = []


    for row in range(ACdf_len):
        if pd.isnull(ACdf['OffsetX'][row]):
            ideality = np.NaN
            idealness.append(ideality)
        else:
            offset = abs(ACdf['OffsetX'][row])
            if 1>offset>0:
                ideality = "Ideal"
                idealness.append(ideality)

            elif 3>offset>1:
                ideality = "Okay"
                idealness.append(ideality)

            elif offset>3:
                ideality = "NOT Ideal"
                idealness.append(ideality)

    ACdf['IdealnessX'] = idealness

    ACdf.to_csv("All_Data.csv", index = False)

    ACdf = ACdf[['left_x','left_y','Z_rw', 'OffsetX','Y_LP','Euclidean_Z']]
    ACdf.to_csv("Floris_Data.csv", index = False)

#calibrate('Remote_Dataset_2_2021-06-24_11-27')
#rectification(2,'Unrectified_images/lft_cam_preview.jpg','Unrectified_images/rht_cam_preview.jpg','In_&_Ex_parameters/Calib_param_B1.5_13:40:46_sunlit.npz')
#marker_selection('Marker_L_rect.png','Marker_R_rect.png', 'Calib_param_B1.5_13:40:46_sunlit.npz')
#triangulate_aircraft("coordinates.csv",'output/Marker_Depth.csv','In_&_Ex_parameters/Calib_param_B1.5_13:40:46_sunlit.npz')
#read_in_ex_param('/home/laptop/stereo-vision-tracker/input/Calib_param_B1.5_13:40:46_sunlit.npz')
