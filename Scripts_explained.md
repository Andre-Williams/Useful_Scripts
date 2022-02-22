# Stereo Vision

## Calibration

- The *Calibration.py* script takes in the calibration image pairs as input and uses them to get the intrinsic parameters of each camera i.e. the camera matrix and the distortion coefficients. 

- These parameters are then used to calibrate each of the cameras thus producing a new set of intrinsic paramters.

- These new set of intrinsic parameters for each camera are then used as input to calibrate the stereo camera system producing another set of intrinsic and extrinsic parameters.

- This final set intrinsic and extrinsic parameters are then saved into a *.npz* file for use in the rectification process.

- Only the *Intrinsic_&_Extrinsic_parameters.npz* file has been uploaded and not the calibration images.

## Rectification

- The *Rectification.py* script reads in the paramters from the *Intrinsic_&_Extrinsic_parameters.npz* file.

- The *Rectification.py* script takes in the image pair/s that you want to be rectified using the intrinsic and extrinsic parameters read in.

- The output of this script is a rectified image pair. 

- I have added 2 image pairs that are unrectified to the repo that can be used as input to test the script. These image pairs are located in the *Unrectified_images* folder.


## Disparity Map

- The *Disparity_map.py* script takes rectified image pairs as input and creates disparity maps/images from them.

- The script is currenty set up to:
  - Output a disparity map based on the input image pair.
  
  - Display the disparity map in a pop-up window which you can double click on to extract the disparity value at the clicked pixel co-ordinates. Once you double click the image the disparity value at the clicked co-ordinates is converted to a depth/distance value which is then displayed in the terminal.
  
    - The disparity to depth/distance conversion is done using a regression formula which can be viewed in the *coords_mouse_disp()* function in the script. **N.B The regression formula currently in the script is one created for the stereo camera system used in the main tutorial I followed. Each stereo camera system will have a unique formula.**
      
      **However, the plan should be to utilise epipolar geometry techniques to do the disparity to depth/distance conversion and not regression.**

- There are 2 sample image pairs that were found online that have already been rectified correctly in the Rectified_image/Samples folder. These image pairs can be used as input for this script. Alternatively, the rectified image pairs produced from the Rectification.py script can also be used.  

## Current Problems

The disparity maps produced using the image pairs rectified using the Rectification.py script are not appearing as they should be i.e. some objects in these disparty maps that are near are being classified as far and some objects that are further away or classified as near.

This problem does not, however, seem to occur when producing the disparity maps using the sample images that I found online that have already been rectified.

Therefore, I suspect that there is something in the Calibration.py or Rectification.py script that I have set up incorrectly.

## Resources
**Main stereo vision tutorial I followed:** 
- Youtube video [link](https://www.youtube.com/watch?v=xjx4mbZXaNc).
- Github [link](https://github.com/LearnTechWithUs/Stereo-Vision)

**OpenCV Camera Calibration and 3D Reconstruction Tutorials:**
- Provides theory and basic python code. [link](https://docs.opencv.org/master/d9/db7/tutorial_py_table_of_contents_calib3d.html)

**Another stereo vision tutorial series**
- Explains parameters and outputs better than previous tutorials.
  - OpenCV Camera Calibration. [link](https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844)
  - Camera and stereo camera calibration. [link](https://python.plainenglish.io/the-depth-i-stereo-calibration-and-rectification-24da7b0fb1e0)
  - Disparity Map. [link](https://python.plainenglish.io/the-depth-ii-block-matching-d599e9372712)

**Stereo Vision OpenCV Python Library**
- For descriptions of stereo vision functions and their parameters. [link](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5) 
