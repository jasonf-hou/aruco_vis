'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
from scipy.spatial.transform import Rotation as R

import math
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    #parameters = cv2.aruco.DetectorParameters_create()
    h, w, _ = frame.shape
    width=500
    height = int(width*(h/w))
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(cv2.aruco_dict, parameters)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = detector.detectMarkers(frame)
    #objp = np.zeros((6*7,3), np.float32)
    #objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    objp = np.load("caldata/objp.npy")
    objp = objp.astype('float32')
    rvec = np.load("caldata/rvecs.npy")
    tvec = np.load("caldata/tvecs.npy")
    rvec2 = rvec
    tvec2 = tvec
    obj_points = np.array([[0.0, 0.0, 0.0], [0.192, 0.0, 0.0], [0.192, 0.12, 0.0], [0.0, 0.12, 0.0]])
    coords = []
    angles = []

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            corner = corners[i].astype('float32')
            (topLeft, topRight, bottomRight, bottomLeft) = corners[0][0][0],corners[0][0][1],corners[0][0][2],corners[0][0][3]
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            retval, rvec2, tvec2 = cv2.solvePnP(obj_points, corner, matrix_coefficients, distortion_coefficients, rvec[i], tvec[i], True)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            #if topLeft[1]!=topRight[1] or topLeft[0]!=bottomLeft[0]:
            #    rot1=np.degrees(np.arctan((topLeft[0]-bottomLeft[0])/(bottomLeft[1]-topLeft[1])))
            #    rot2=np.degrees(np.arctan((topRight[1]-topLeft[1])/(topRight[0]-topLeft[0])))
            #    rot=(np.round(rot1,3)+np.round(rot2,3))/2
            #    print(rot1,rot2,rot)
            #else:
            #    rot=0 
            #rotS="rotation:"+str(np.round(rot,3))
            #print(rotS)
            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec2, tvec2, 0.2)
            # Print the pose for the ArUco marker
            # The pose of the marker is with respect to the camera lens frame.
            # Imagine you are looking through the camera viewfinder, 
            # the camera lens frame's:
            # x-axis points to the right
            # y-axis points straight down towards your toes
            # z-axis points straight ahead away from your eye, out of the camera
            #for i, marker_id in enumerate(marker_ids):
            
            # Store the translation (i.e. position) information
            #print(tvec2)
            #print((tvec2[0][0], tvec2[1][0], tvec2[2][0]))
            coords.append((tvec2[0][0], tvec2[1][0], tvec2[2][0])) 
            #print(np.linalg.norm(tvec2))

            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3], _ = cv2.Rodrigues(rvec2)
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()   
                
            # Quaternion format     
            transform_rotation_x = quat[0] 
            transform_rotation_y = quat[1] 
            transform_rotation_z = quat[2] 
            transform_rotation_w = quat[3] 
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,transform_rotation_y,transform_rotation_z,transform_rotation_w)
             
            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            print((roll_x, pitch_y, yaw_z))
            angles.append((roll_x, pitch_y, yaw_z))


    return frame, (angles,coords)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    angles = np.zeros((6899,3))
    coords = np.zeros((6899,3))

    video = cv2.VideoCapture('IMG_1537.mp4')#cv2.VideoCapture(0)
    time.sleep(2.0)
    count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output, metadata = pose_esitmation(frame, aruco_dict_type, k, d)
        angle = metadata[0]
        coord = metadata[1]
        print(angle)
        if len(angle) == 0 or len(coord) == 0:
            angles[count, 0] = angles[count-1, 0]
            angles[count, 1] = angles[count-1, 1]
            angles[count, 2] = angles[count-1, 2]
        #elif len(coord) == 0:
            coords[count, 0] = coords[count-1, 0]
            coords[count, 1] = coords[count-1, 1]
            coords[count, 2] = coords[count-1, 2]
        else:
            angles[count, 0] = angle[0][0]
            angles[count, 1] = angle[0][1]
            angles[count, 2] = angle[0][2]
            coords[count, 0] = coord[0][0]
            coords[count, 1] = coord[0][1]
            coords[count, 2] = coord[0][2]
        cv2.imshow('Estimated Pose', output)
        count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    np.save("angles.npy", angles)
    np.save("coords.npy", coords)
    video.release()
    cv2.destroyAllWindows()