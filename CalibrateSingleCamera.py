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
from DetectCorner import *
BOARDHEIGHT = 27
BOARDWIDTH = 27

def drawMap(cur_map, im, showText = False):
    if len(im.shape)<3:
        new_im =  np.stack((im,)*3, axis=-1)
    else:
        new_im = np.copy(im)
    for i in range(cur_map.shape[0]):
        for j in range(cur_map.shape[1]):
            pt = cur_map[i,j,:2]
            pt = pt.astype(np.int32).tolist()
            if pt[0]>0 and pt[1]>0:
                if showText is True:
                    str1 = '('+str(i)+ ','+str(j)+')'
                    new_im = cv2.putText(new_im, str1 ,(pt[0],pt[1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                     fontScale=.5,color=(0,0,255))
                new_im = cv2.drawMarker(new_im, (pt[0],pt[1]),(0,0,255), markerType=cv2.MARKER_CROSS, \
                                    markerSize=4, thickness=1, line_type=cv2.LINE_AA)
    return new_im

def drawMapMarker(cur_map, im, marker_color = (0,0,255), marker_type = cv2.MARKER_STAR ):
    new_im = np.copy(im)
    for i in range(cur_map.shape[0]):
        for j in range(cur_map.shape[1]):
            pt = cur_map[i,j,:2]
            pt = pt.astype(np.int32).tolist()
            if pt[0]>0 and pt[1]>0:
                new_im = cv2.drawMarker(new_im, (pt[0],pt[1]),marker_color, markerType=marker_type, \
                                    markerSize=4, thickness=1, line_type=cv2.LINE_AA)
    return new_im

# find the top left corner of all images and set them to (0,0), this only works for perfect calibration board
def alignCornerMap(map_list):
    num_img = len(map_list)

    # find top left corner 
    for idx in range(num_img):
        cur_map = map_list[idx]
        start_col=0
        for i in range(cur_map.shape[0]):
            counter = 0 
            for j in range(cur_map.shape[1]):
                counter+=cur_map[i,j,0]
            if counter>-cur_map.shape[1]+.1:
                start_col = i
                break
        start_row=0
        for j in range(cur_map.shape[1]):
            counter = 0 
            for i in range(cur_map.shape[0]):
                counter+=cur_map[i,j,0]
            if counter>-cur_map.shape[0]+.1:
                start_row = j
                break        
        new_map = np.roll(cur_map, -start_col, axis = 0) 
        map_list[idx] = np.roll(new_map, -start_row, axis = 1) 
    return map_list

# Find the corners exist in all images
def collectCorners(map_list):
    num_img = len(map_list)
    validIdx = []
    img_points = []
    obj_points= []
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            validpoint = True
            for idx in range(num_img): 
                if map_list[idx][i,j,0]<0:
                    validpoint = False
                    break
            if validpoint is True:
                validIdx.append(np.array([i,j]))
    for idx in range(num_img):
        validpoints = []
        valid3Dpoints = []
        for i in range(len(validIdx)):
            cur_corner = map_list[idx][validIdx[i][0],validIdx[i][1],:]
            validpoints.append(cur_corner)
            valid3Dpoints.append(np.array([validIdx[i][0],validIdx[i][1],0])*10) #10mm board
        tmp_p = np.expand_dims(np.array(validpoints),axis=1).astype(np.float32)
        img_points.append(tmp_p)
        obj_points.append(np.array(valid3Dpoints).astype(np.float32))
    return img_points, obj_points

def calibrateSingleCamera(calib_files, show_plot=True):
    img_list = []
    map_list = []
    for i in range(len(calib_files)):
        print(calib_files[i])
        im=cv2.imread(calib_files[i])
        # corner_im = np.copy(im)
        im2=255-im[:,:,1]
        ret,draw_quads,draw_longquads,final_map, org_map = detectCorner(im2,frontDetect = False,allowTwoPointContour = True)
        if ret is True:
            img_list.append(im)
            map_list.append(final_map)
            if show_plot is True:
                im_p = im.copy()
                im_p = cv2.drawContours(im_p, draw_quads, -1, (255, 0, 0), 3)
                im_p = cv2.drawContours(im_p, draw_longquads, -1, (0, 0, 255), 3)
                cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
                cv2.imshow('contour', im_p)

                im_new = drawMap(final_map, im, False)
                cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
                cv2.imshow('dst', im_new)
                name = 'dump/Grid_{}.png'.format(i + 1)
                cv2.imwrite(name, im)
                key = cv2.waitKey(1000)
                if key == 27:
                    cv2.destroyAllWindows()
    map_list = alignCornerMap(map_list)
    img_points, obj_points = collectCorners(map_list)
    
    #
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

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im2.shape[::-1],None,None)  
    print(ret) 
    print(mtx)
    print(dist) 
    return ret,mtx, dist, rvecs, tvecs, img_points, obj_points      

def get_homography(k_matrix, r_vector, t_vector,zoom_factor,resize_factor,im_sz):

    r_matrix, _ = cv2.Rodrigues(r_vector)
    hr = np.dot(k_matrix, np.dot(np.transpose(r_matrix), np.linalg.inv(k_matrix)))
    c = -np.dot(np.transpose(r_matrix), t_vector)
    u0 = -np.dot(k_matrix, c) / c[2]
    u = np.zeros((3, 1))

    u[0] = u0[0]
    u[1] = u0[1]
    u[2] = -u0[2]

    ht = np.array([[1, 0], [0, 1], [0, 0]])
    ht = np.hstack((ht, u))
    h = np.dot(ht, hr)
    scale=np.array([[resize_factor, 0,0], [0, resize_factor,0], [0, 0, 1]])
    h = np.dot(scale, h)
    cx = im_sz[1]*resize_factor/2
    cy = im_sz[0]*resize_factor/2
    zoom = np.array([[zoom_factor, 0,(1-zoom_factor)*cx], [0, zoom_factor,(1-zoom_factor)*cy], [0, 0, 1]])
    # zoom = np.array([[1, 0,100], [0, 1,100], [0, 0, 1]])
    h = np.dot(zoom, h)
    return h

def get_homography_2D(im_point, obj_point,im_sz):
    
    block_sz = min(im_sz[0],im_sz[1])
    p2d = np.squeeze(im_point)
    p3d = np.copy(p2d)
    for i in range(len(obj_point)):
        p3d[i,0] = obj_point[i,0]/BOARDWIDTH*im_sz[1]
        p3d[i,1] = obj_point[i,1]/BOARDHEIGHT*im_sz[0]
    h,mask = cv2.findHomography(p2d, p3d)
   
    return h

def transferMap(final_map, H):
    orgpt=[]
    for i in range(final_map.shape[0]):
        for j in range(final_map.shape[1]):
            if final_map[i,j,0]>=0:
                orgpt.append(final_map[i,j,:])
    newpt = np.squeeze(cv2.perspectiveTransform(np.array([orgpt]), H))
    idx =0
    for i in range(final_map.shape[0]):
        for j in range(final_map.shape[1]):
            if final_map[i,j,0]>=0:
                final_map[i,j,:] = newpt[idx]
                idx+=1

def saveParams(fname,K,D,err):
    # camera_matrix = K
    # dist_coeff = D
    # proj_err = err
    data = {"camera_matrix": K.tolist(), "dist_coeff": D.tolist(), "re-projection_err": err}
    with open(fname, "w") as f:
        yaml.dump(data, f)


save_params = True
calib_base_name = "calib_left"
res_folder = 'dump'
iteration =1
zoom_factor = 0.8
resize_factor = 1.0
show_plot = True
#walk though all files in folder
calib_files = []

calib_img_path = 'D:/work/NIST/data/new_data/calib_0829/single/left'
# for dirpath, dnames, fnames in os.walk(calib_img_path):
#     for f in fnames:
#         if f.endswith(".tif") or f.endswith(".bmp"):
#             calib_files.append(os.path.join(dirpath, f))
for f in os.listdir(calib_img_path):
    if f.endswith(".tif") or f.endswith(".bmp") or f.endswith(".jpg") or f.endswith(".png"):
        calib_files.append(os.path.join(calib_img_path, f))
ret,K, D, rvecs, tvecs,img_points,obj_points = calibrateSingleCamera(calib_files,show_plot=True)
if save_params:
    saveParams(calib_base_name+"_iter_1.yaml", K, D, ret)


#img_list = []
map_list = []
for i in range(len(calib_files)):
    my_image = cv2.imread(calib_files[i])
    print(calib_files[i])
    new_im = 255-my_image[:,:,0]
    # undist = cv2.undistort(new_im, K, D)
    # name = res_folder + '/ID{}UndistortIT{}.png'.format(i + 1, iteration)
    # cv2.imwrite(name, my_image)

    H = get_homography(K, rvecs[i], tvecs[i],zoom_factor,resize_factor,my_image.shape)
    # H = get_homography_2D(img_points[i], obj_points[i],(my_image.shape[0]*resize_factor,my_image.shape[1]*resize_factor))
#    result = cv2.warpPerspective(undist, H, (int(my_image.shape[1]*resize_factor),int(my_image.shape[0]*resize_factor)), flags=cv2.INTER_LINEAR,\
#                                 borderMode = cv2.BORDER_CONSTANT, borderValue = (255,255,255) )
    result = cv2.warpPerspective(new_im, H, (int(my_image.shape[1]*resize_factor),int(my_image.shape[0]*resize_factor)), flags=cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE )
    # For each image, we take the four "out" corners
    # Then we re-project them on a perfect mapping

    name = res_folder + '/ID{}UnprojectIT{}.png'.format(i + 1, iteration)
    cv2.imwrite(name, result)




    im2=result
    r, draw_quads,draw_longquads,final_map, org_map = detectCorner(im2, frontDetect = False,allowTwoPointContour = True)
    
    if show_plot is True:
        im_p =  np.stack((result,)*3, axis=-1)
        im=cv2.drawContours(im_p, draw_quads,-1, (255,0,0), 3)
        im=cv2.drawContours(im, draw_longquads,-1, (0,0,255), 3)
        cv2.namedWindow('contour',cv2.WINDOW_NORMAL)
        cv2.imshow('contour',im)
        im = drawMap(final_map, im_p)
        cv2.namedWindow('dst_rect',cv2.WINDOW_NORMAL)
        cv2.imshow('dst_rect',im)
        key = cv2.waitKey(1000)#pauses for 1 seconds before fetching next image
        # if key == 27:#if ESC is pressed, exit loop
            # cv2.destroyAllWindows()
    H_inverse = np.linalg.inv(H)
    transferMap(final_map, H_inverse)
    if show_plot is True:
        im_p =  my_image.copy()
        im = drawMap(final_map, im_p)
        cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
        cv2.imshow('dst',im)
        key = cv2.waitKey(1000)#pauses for 1 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
#    img_list.append(im)
    map_list.append(final_map)
    print(i)
        
map_list = alignCornerMap(map_list)
img_points, obj_points = collectCorners(map_list)
        
        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im2.shape[::-1],None,None)
if save_params:
    saveParams(calib_base_name+"_iter_2.yaml", K, D, ret)
print(ret) 
print(mtx)
print(dist) 
cv2.destroyAllWindows()        


