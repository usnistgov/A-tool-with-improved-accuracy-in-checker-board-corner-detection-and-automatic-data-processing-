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



import cv2 as cv
import numpy as np


# Find corners to remap the image to canonical pattern
def canonique(fname, corner_qty, criteria): 
    img = cv.imread(fname)
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = grayscale.shape[::-1]
    ret, old_corners = cv.findChessboardCorners(grayscale, corner_qty, None)

    if ret is True:
        cv.cornerSubPix(grayscale, old_corners, (11, 11), (-1, -1), criteria)
        old_corners = np.asarray(old_corners).reshape(-1, 2)
        temp = np.zeros((old_corners.shape[0], 1))
        old_corners = np.hstack((old_corners, temp))
        old_corners = np.asarray(old_corners).astype('float32')

        af_trans = []
        width_step = float(image_size[0]) / float(corner_qty[0] + 1)
        height_step = float(image_size[1]) / float(corner_qty[1] + 1)
        af_trans.append([width_step, height_step])
        af_trans.append([image_size[0] - width_step, height_step])
        af_trans.append([width_step, image_size[1] - height_step])
        af_trans.append([image_size[0] - width_step, image_size[1] - height_step])
        af_trans = np.asarray(af_trans).astype('float32')

        bf_trans = [old_corners[0, 0:2],                                       # top-left point
                    old_corners[corner_qty[0] - 1, 0:2],                       # top-right point
                    old_corners[(corner_qty[1] - 1) * corner_qty[0], 0:2],     # bottom-left point
                    old_corners[corner_qty[1] * corner_qty[0] - 1, 0:2]]       # bottom-right point
        bf_trans = np.asarray(bf_trans).astype('float32')

        trans = cv.getPerspectiveTransform(bf_trans, af_trans)
        result = cv.warpPerspective(grayscale, trans, image_size)
        cv.imwrite(fname, result)


# Find corners in the list of images
def find_corners(filename, criteria, ctrl, corner_size, export_result=False, base_folder=''):
    image_size_read = (0, 0)
    control_pts, projected_pts = [], []

    for index, fname in enumerate(filename):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image_size_read = gray.shape[::-1]
        ret_find, corners = cv.findChessboardCorners(gray, corner_size, None)

        if ret_find is True:
            control_pts.append(ctrl)
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            projected_pts.append(corners)

            if export_result is True:
                cv.drawChessboardCorners(img, corner_size, corners, ret_find)
                cv.imwrite(base_folder + '/ID{}FindCorner.png'.format(index + 1), img)

    return image_size_read, np.asarray(control_pts), projected_pts


def get_homography(k_matrix, r_vector, t_vector):
    r_matrix, _ = cv.Rodrigues(r_vector)
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

    return h


def normalize_control_points(ctrl_pts, image_res, corner_qty, norm_length):
    # Process the list of control points in each image
    for item, pt in enumerate(ctrl_pts):
        # the points are organized in the list by n_columns of n_rows points
        # We have
        #   corner_size[0] = n_rows
        #   corner_size[1] = n_columns
        pt[:, 0] *= float(corner_qty[0] + 1) * norm_length / image_res[0]
        pt[:, 1] *= float(corner_qty[1] + 1) * norm_length / image_res[1]
        ctrl_pts[item] = pt

    return ctrl_pts


corner_size = (9, 6)  # Quantity of corners on the checker board
corner_length = 1.0
iteration_quantity = 20
image_quantity = 9
res_folder = './Results'

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0), ...
control = np.zeros((corner_size[0] * corner_size[1], 3), np.float32)
control[:, :2] = np.mgrid[1: corner_size[0] + 1,
                          1: corner_size[1] + 1].T.reshape(-1, 2) * corner_length

TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
file = []

# There should be image_quantity files for the camera
for i in range(image_quantity):
    file.append('./CheckerBoard-Images/checker_board_{:04d}_right.png'.format(i + 1))

# Process all of them to retrieve the checker boards corners.
# This is the initialization step of the iterative algorithm
image_size, control_points, projected_points = find_corners(file,
                                                            TERMINATION_CRITERIA,
                                                            control,
                                                            corner_size,
                                                            True,
                                                            res_folder)

control_points = np.asarray(control_points).astype('float32')
projected_points = np.asarray(projected_points).astype('float32')

rms, K, D, rvec, T = cv.calibrateCamera(control_points,
                                        projected_points,
                                        image_size,
                                        None, None)
rvec = np.asarray(rvec)
T = np.asarray(T)

# Compute the re-projection error to serve as baseline for further computation
print('Initialization Step')
print('Calibration RMS= ', rms)
print('Fx={}, Fy={}, Cx={}, Cy={}, Dk1={}, Dk2={}'.format(K[0, 0],
                                                          K[1, 1],
                                                          K[0, 2],
                                                          K[1, 2],
                                                          D[0, 0],
                                                          D[0, 1]))

# From the research paper "Accurate Camera Calibration using Iterative
# Refinement of Control Points", the following steps are
# LOOP START
# 1/ Un-distort and un-project: Use M/D/R/T to project the input images
#                               to a canonical pattern after removing distortions
# 2/ Localize control points: Localize calibration pattern control points
#                             in the canonical pattern
#
# 3/ Re-project: Project the control points using the estimated camera parameters
# 4/ Parameter Fitting: Use the projected control points to refine the camera
#                       parameters using Levenberg-Marquardt
# Note: THE CONVERGENCE CRITERIA IS NOT YET DEFINED

for iteration in range(iteration_quantity):
    iter_filename = []

    ################################################################################################
    # 1/ Un-distort and un-project
    for i in range(image_quantity):
        my_image = cv.imread(file[i])
        undist = cv.undistort(my_image, K, D)
        name = res_folder + '/ID{}UndistortIT{}.png'.format(i + 1, iteration)
        cv.imwrite(name, undist)

        H = get_homography(K, rvec[i], T[i])
        result = cv.warpPerspective(undist, H, (1280, 720), flags=cv.INTER_LINEAR)

        # For each image, we take the four "out" corners
        # Then we re-project them on a perfect mapping

        name = res_folder + '/ID{}UnprojectIT{}.png'.format(i + 1, iteration)
        cv.imwrite(name, result)
        iter_filename.append(name)
        canonique(name, corner_size, TERMINATION_CRITERIA)

    ################################################################################################
    # 2a/ Find the location of the new control points
    iter_ctrl_points = []
    iter_proj_points = []

    for index, fname in enumerate(iter_filename):
        img = cv.imread(fname)
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image_size_read = grayscale.shape[::-1]
        ret, corners = cv.findChessboardCorners(grayscale, corner_size, None)

        if ret is True:
            cv.cornerSubPix(grayscale, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
            cv.drawChessboardCorners(img, corner_size, corners, ret)
            cv.imwrite(res_folder + '/ID{}NewpointsIT{}.png'.format(index + 1, iteration), img)

            corners = np.asarray(corners).reshape(-1, 2)
            temp = np.zeros((corners.shape[0], 1))
            corners = np.hstack((corners, temp))
        else:
            print('Error (It={}) on image {}'.format(iteration + 1, index + 1))

        iter_ctrl_points.append(corners)

    iter_ctrl_points = np.asarray(iter_ctrl_points).astype('float32')

    # 2b/ Normalization of the control points
    iter_ctrl_points = normalize_control_points(iter_ctrl_points,
                                                image_size,
                                                corner_size,
                                                corner_length)

    ################################################################################################
    # 3/ Project the new control points
    for i in range(len(iter_ctrl_points)):
        proj_points, _ = cv.projectPoints(iter_ctrl_points[i], rvec[i], T[i], K, D)
        iter_proj_points.append(proj_points)

    ################################################################################################
    # 4/ Parameter fitting (new camera calibration)
    rms, K, D, rvec, T = cv.calibrateCamera(control_points,
                                            iter_proj_points,
                                            image_size, K, D)

    rvec = np.asarray(rvec)
    T = np.asarray(T)
    print('Iterative Step {}'.format(iteration + 1))
    print('Calibration RMS= ', rms)
    print('Fx={}, Fy={}, Cx={}, Cy={}, Dk1={}, Dk2={}'.format(K[0, 0],
                                                              K[1, 1],
                                                              K[0, 2],
                                                              K[1, 2],
                                                              D[0, 0],
                                                              D[0, 1]))
