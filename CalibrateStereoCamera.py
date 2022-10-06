# license statement

NIST-developed software is provided by NIST as a public service. You may use, copy, and distribute copies of the software in any medium,
provided that you keep intact this entire notice. You may improve, modify, and create derivative works of the software or any portion of
the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed
the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards
and Technology as the source of the software. 
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY
OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT,
AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY
DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF,
INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated
with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data,
programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where
a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection
within the United States.



import cv2
import matplotlib.pyplot as plt
#%matplotlib qt
import numpy as np
import math
import os
import yaml
from DetectCorner import detectCorner
BOARDHEIGHT = 27
BOARDWIDTH = 27



# Find the corners exist in all images
def collectCorners(map_left_list,map_right_list):
    assert(len(map_left_list)==len(map_right_list))
    num_img = len(map_left_list)
    validIdx = []
    img_left_points = []
    img_right_points = []
    obj_points= []
    #remove those bad images
    good_left = []
    good_right = []
    for idx in range(num_img):
        numGood = 0
        for i in range(BOARDHEIGHT):
            for j in range(BOARDWIDTH):
                if map_left_list[idx][i,j,0]>=0 and map_right_list[idx][i,j,0]>0:
                    numGood +=1
        if numGood >20:
            good_left.append(map_left_list[idx])
            good_right.append(map_right_list[idx])
        else:
            print('remove images due to lack of point' + str(idx) )
    num_img  = len(good_left)
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            validpoint = True
            for idx in range(num_img): 
                if good_left[idx][i,j,0]<0 or good_right[idx][i,j,0]<0 :
                    validpoint = False
                    break
            if validpoint is True:
                validIdx.append(np.array([i,j]))
    for idx in range(num_img):
        validleftpoints = []
        validrightpoints = []
        valid3Dpoints = []
        for i in range(len(validIdx)):
            cur_corner = good_left[idx][validIdx[i][0],validIdx[i][1],:]
            validleftpoints.append(cur_corner)
            cur_corner = good_right[idx][validIdx[i][0],validIdx[i][1],:]
            validrightpoints.append(cur_corner)
            valid3Dpoints.append(np.array([validIdx[i][0],validIdx[i][1],0])*10) #board size 10mm
        tmp_p = np.expand_dims(np.array(validleftpoints),axis=1).astype(np.float32)
        img_left_points.append(tmp_p)
        tmp_p = np.expand_dims(np.array(validrightpoints),axis=1).astype(np.float32)
        img_right_points.append(tmp_p)
        obj_points.append(np.array(valid3Dpoints).astype(np.float32))
    return img_left_points, img_right_points, obj_points

def calibrateStereoCamera(calib_left_files, calib_right_files,KK1,D1,KK2,D2,show_plot=True):
    img_left_list = []
    map_left_list = []
    img_right_list = []
    map_right_list = []
    im = cv2.imread(calib_left_files[0])
    for i in range(len(calib_left_files)):
        print(calib_left_files[i])
        print(calib_right_files[i])
        im1=cv2.imread(calib_left_files[i])
        im1l=255-im1[:,:,1]
        ret1,draw_quads,draw_longquads,final_map1, org_map1 = detectCorner(im1l,frontDetect = False,allowTwoPointContour = True)
        if show_plot is True:
            for k in range(len(draw_quads)):
                im=cv2.drawContours(im1, draw_quads,k , (255,0,0), 3)
            im=cv2.drawContours(im, draw_longquads, -1 , (0,0,255), 3)
            cv2.namedWindow('left',cv2.WINDOW_NORMAL)
            cv2.imshow('left',im)
        im2=cv2.imread(calib_right_files[i])
        im2l=255-im2[:,:,1]
        ret2,draw_quads,draw_longquads,final_map2, org_map2 = detectCorner(im2l,frontDetect = False,allowTwoPointContour = True)
        if show_plot is True:
            for k in range(len(draw_quads)):
                im=cv2.drawContours(im2, draw_quads,k , (255,0,0), 3)
            im=cv2.drawContours(im, draw_longquads, -1 , (0,0,255), 3)
            cv2.namedWindow('right',cv2.WINDOW_NORMAL)
            cv2.imshow('right',im)
            key = cv2.waitKey(10)

        if ret1 is True and ret2 is True:
            img_left_list.append(im1)
            map_left_list.append(final_map1)
            img_right_list.append(im2)
            map_right_list.append(final_map2)


    w=im.shape[1]
    h=im.shape[0]
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                            cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    img_left_points, img_right_points, obj_points = collectCorners(map_left_list,map_right_list)
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(obj_points,
                                                         img_left_points,
                                                         img_right_points,
                                                         KK1,D1,
                                                         KK2,D2,
                                                         criteria=stereocalib_criteria,
                                                         flags=flags,
                                                         imageSize=(w,h)
                                                         )
    # #
    # if show_plot is True:
    #     for idx in range(len(map_list)):
    #         final_map = map_list[idx]
    #         im = drawMap(final_map, img_list[idx], False)
    #         cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
    #         cv2.imshow('dst',im)
    #         name =  'dump/Grid_{}.png'.format(idx + 1)
    #         cv2.imwrite(name,im)
    #         key = cv2.waitKey(1000)#pauses for 1 seconds before fetching next image
    #         if key == 27:#if ESC is pressed, exit loop
    #             cv2.destroyAllWindows()
    #
    #     cv2.destroyAllWindows()
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im2.shape[::-1],None,None)
    # print(ret)
    # print(mtx)
    # print(dist)
    print([ret, M1, d1, M2, d2, R, T, E, F ])
    return ret, M1, d1, M2, d2, R, T, E, F



def saveParams(fname,KK1,D1,KK2,D2,R,T,E,F,err):
    # camera_matrix = K
    # dist_coeff = D
    # proj_err = err
    data = {"camera_matrix_left": KK1.tolist(),
            "camera_matrix_right": KK2.tolist(),
            "dist_coeff_left": D1.tolist(),
            "dist_coeff_right": D2.tolist(),
            "Rotation": R.tolist(),
            "Translation": T.tolist(),
            "essential_matrix": E.tolist(),
            "fundamental_matrix": E.tolist(),
            "re-projection_err": err}
    with open(fname, "w") as f:
        yaml.dump(data, f)
    fname = fname.replace('yaml', 'xml')
    cv_file = cv2.FileStorage(fname, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix_left", KK1)
    cv_file.write("camera_matrix_right", KK2)
    cv_file.write("dist_coeff_left", D1)
    cv_file.write("dist_coeff_right", D2)
    cv_file.write("Rotation", R)
    cv_file.write("Translation", T)
    cv_file.write("essential_matrix", E)
    cv_file.write("fundamental_matrix", F)
    cv_file.write("projection_err", err)
    cv_file.release()

save_params = True

res_folder = 'dump'
iteration =1
zoom_factor = 0.8
resize_factor = 1.0
show_plot = True
#walk though all files in folder
calib_left_files = []
calib_left_img_path = 'D:/work/NIST/data/new_data/calib_0829/stereo/left/'

calib_right_files = []
calib_right_img_path = 'D:/work/NIST/data/new_data/calib_0829/stereo/right/'

with open('D:/work/NIST/src/python/calib_left_iter_1.yaml') as f:
    cam_l = yaml.load(f, Loader=yaml.FullLoader)
with open('calib_right_iter_1.yaml') as f:
    cam_r = yaml.load(f, Loader=yaml.FullLoader)

KK1 = np.array(cam_l['camera_matrix'])
D1 = np.array(cam_l['dist_coeff'])
KK2 = np.array(cam_r['camera_matrix'])
D2 = np.array(cam_r['dist_coeff'])

for f in os.listdir(calib_left_img_path):
    if f.endswith(".tif") or f.endswith(".bmp") or f.endswith(".jpg") or f.endswith(".png"):
        calib_left_files.append(os.path.join(calib_left_img_path, f))
for f in os.listdir(calib_right_img_path):
    if f.endswith(".tif") or f.endswith(".bmp") or f.endswith(".jpg") or f.endswith(".png"):
        calib_right_files.append(os.path.join(calib_right_img_path, f))
def findLastNuminStr(str):
    val=[int(s) for s in str.split() if s.isdigit()]
    return(val[-1])

# calib_left_files=sorted(calib_left_files, key = findLastNuminStr)
# calib_right_files=sorted(calib_right_files, key = findLastNuminStr)
# calib_left_files = calib_left_files[:16]
# calib_right_files = calib_right_files[:16]
ret, M1, d1, M2, d2, R, T, E, F = calibrateStereoCamera(calib_left_files, calib_right_files,KK1,D1,KK2,D2,show_plot=True)

print([ret, M1, d1, M2, d2, R, T, E, F])
saveParams("stereo_calibration.yaml",M1,d1,M2,d2,R,T,E,F,ret)

