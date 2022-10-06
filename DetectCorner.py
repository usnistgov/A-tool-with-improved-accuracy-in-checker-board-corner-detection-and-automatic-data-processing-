
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
import matplotlib
# matplotlib.use('tkagg', warn=False)
import numpy as np
import math
from scipy.spatial import distance as dist
MAX_CONTOUR_APPROX=20
BOARDHEIGHT = 27
BOARDWIDTH = 27
MIN_AREA=150

def generateQuads(img, frontalDetect = False, min_size=None, allowTwoPointContour= False):
        
    if allowTwoPointContour is True:
        MIN_AREA = 100
    if min_size is None:
        min_size = img.shape[0]*img.shape[1]/30/30*0.01

    filterQuads = True

    contours, hierarchy = cv2.findContours(img,  cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    
    contour_quads=[]
    contour_parent =[]
    contour_long_quads = []
    contour_long_parent =[]
    contour_child_counter = np.zeros(len(contours),)
    boardIdx = -1
    bad_contours = []
    for idx in range(len(contours)):

        tmp = contours[idx]
        dis =1e9
        for i in range(len(tmp)):
            tmpDis = np.linalg.norm(tmp[i,0,:]-np.array([674,641]))
            if tmpDis <dis:
                dis  = tmpDis
        if dis<10:
            a=1
        parentIdx = hierarchy[0,idx,3]
        if (hierarchy[0,idx,2] != -1 or parentIdx == -1):
            bad_contours.append(contours[idx])
            continue

        if (hierarchy[0,idx,2] != -1 ):
            continue
        contour = contours[idx]
        contour_rect = cv2.boundingRect(contour);
        if (contour_rect[2]*contour_rect[3]< min_size):  
            continue
        
        for approx_level in range(MAX_CONTOUR_APPROX):
            approx_contour= cv2.approxPolyDP(contours[idx], approx_level+1, True)
            if approx_contour.shape[0]==4:
                break
            approx_contour= cv2.approxPolyDP(approx_contour, approx_level+1, True)
            if approx_contour.shape[0]==4:
                break
            if approx_contour.shape[0] <4:
                break
        if approx_contour.shape[0]!=4:
            if allowTwoPointContour is False:
                continue
            else:
                # in stereo cases, since theboard is small in FOV, the contour rect could reduce to 2
                if approx_contour.shape[0] != 2:
                    continue
                else:
                    # we expend two point contour
                    p1 = approx_contour[0,0,:]
                    p2 = approx_contour[1,0,:]
                    approx_contour = np.concatenate((approx_contour, approx_contour))
                    if abs(p1[0]-p2[0]) < abs(p1[1]-p2[1]):
                        approx_contour[0, 0, :] = np.array([p1[0] - 3, p1[1]])
                        approx_contour[1, 0, :] = np.array([p1[0] + 3, p1[1]])
                        approx_contour[2, 0, :] = np.array([p2[0] + 3, p2[1]])
                        approx_contour[3, 0, :] = np.array([p2[0] - 3, p2[1]])
                    else:
                        approx_contour[0, 0, :] = np.array([p1[0] , p1[1]-3])
                        approx_contour[1, 0, :] = np.array([p1[0] , p1[1]+3])
                        approx_contour[2, 0, :] = np.array([p2[0] , p2[1]+3])
                        approx_contour[3, 0, :] = np.array([p2[0] , p2[1]-3])
        if not cv2.isContourConvex(approx_contour):
            continue
        
        square = 0
        if filterQuads is True:
            p = cv2.arcLength(approx_contour, True)
            area = cv2.contourArea(approx_contour, False)
            d1 = np.linalg.norm(approx_contour[0,0,:]-approx_contour[2,0,:])
            d2 = np.linalg.norm(approx_contour[1,0,:]-approx_contour[3,0,:])
            
            d3 = np.linalg.norm(approx_contour[0,0,:]-approx_contour[1,0,:])
            d4 = np.linalg.norm(approx_contour[1,0,:]-approx_contour[2,0,:])
            
            q = np.squeeze(approx_contour)
            a1,a2,a3,a4 = findAngleOfQuad(q)
#          
            if frontalDetect is False:
                if ( (d3*d4 < area*1.5) and (area > min_size) and (d1 >= 0.15 * p) and (d2 >= 0.15 * p) and (area > 300 )):

                    if (d3*4 > d4 and d4*4 > d3 and a1 <30 and a2 <30) :
                        square = 1
                    elif (d3>3*d4 or d4>3*d3 ) and (a1 <30 or a2 <30) :
                        square = 2
                    else:
                        continue
                else:
                    continue
            else:
                if ( d3*d4 < area*1.5 and area > min_size and\
                    d1 >= 0.15 * p and d2 >= 0.15 * p and area>300 ):
                    if (d3*1.2 > d4 and d4*1.2 > d3 and d1*1.2>d2 and d2*1.2>d1 and a1 <15 and a2 <15 \
                        and abs(a3-90)<20 and abs(a4-90)<20 ): 
                        square = 1
                    elif (d3>3*d4 or d4>3*d3 and a1 <30 and a2 <30 \
                        and abs(a3-90)<45 and abs(a4-90)<45 ) :
                        square = 2
                    else:
                        continue
                else:

                    continue
        
        contour_child_counter[parentIdx]+=1;
        # find the biggest retangle and assign to boardIdx
        if (boardIdx != parentIdx and (boardIdx < 0 or contour_child_counter[boardIdx] < contour_child_counter[parentIdx])):
            boardIdx = parentIdx;
            
        # contour_quads.append(np.squeeze(approx_contour))    
        if square==1:
            contour_quads.append(approx_contour)    
            contour_parent.append(parentIdx)
        elif square ==2:
            contour_long_quads.append(approx_contour)
            contour_long_parent.append(parentIdx)
    # # debug contour
    # contour_im =  np.stack((img,)*3, axis=-1)
    # contour_im=cv2.drawContours(contour_im, contour_quads,-1 , (255,0,0), 3)
    # contour_im = cv2.drawContours(contour_im, contour_long_quads, -1, (0, 0, 255), 3)
    # cv2.namedWindow('contour2',cv2.WINDOW_NORMAL)
    # cv2.imshow('contour2',contour_im)
    # cv2.waitKey()

    ## Found a bug today that snce we use black border, there is a posibility that the long quad will have different
    # parent squards with the rest.  We need to include both long quads in future calibration.
    final_contour = []
    final_long_contour = []
    for idx in range(len(contour_quads)):
        if contour_parent[idx]==boardIdx:
            final_contour.append(contour_quads[idx])
    for idx in range(len(contour_long_quads)):
        if contour_long_parent[idx]==boardIdx:
            final_long_contour.append(contour_long_quads[idx])
    return final_contour,final_long_contour



#COMPUTE INTENSITY HISTOGRAM OF INPUT IMAGE
def GetIntensityHistogram256(img):
    piHist = np.zeros(256,)
    # sum up all pixel in row direction and divide by number of columns
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            piHist[img[j,i]]+=1
    return piHist

#SMOOTH HISTOGRAM USING WINDOW OF SIZE 2*iWidth+1
def SmoothHistogram256( piHist, iWidth = 0):
    if iWidth <0:
        iWidth = 0
    piHistSmooth = np.zeros(256,)    
    for i in range(256):
        iIdx_min = max(0, i - iWidth)
        iIdx_max = min(255, i + iWidth)
        piHistSmooth[i] = np.sum(piHist[iIdx_min:iIdx_max+1])/(iIdx_max -iIdx_min+1)
    return piHistSmooth

#COMPUTE FAST HISTOGRAM GRADIENT
def GradientOfHistogram256(piHist):
    piHistGrad=np.zeros(256,)

    prev_grad = 0;
    for i in range(1,255):
        grad = piHist[i-1] - piHist[i+1];
        if (np.abs(grad) < 100):
            if (prev_grad == 0):
                grad = -100;
            else:
                grad = prev_grad
        piHistGrad[i] = grad;
        prev_grad = grad;
    
    piHistGrad[255] = 0;
    return piHistGrad

#PERFORM SMART IMAGE THRESHOLDING BASED ON ANALYSIS OF INTENSTY HISTOGRAM
def BinarizationHistogramBased(img):

    iCols = img.shape[1]
    iRows = img.shape[0]
    iMaxPix = iCols*iRows;
    iMaxPix1 = iMaxPix/100;
    iNumBins = 256;
    iMaxPos = 20;

    piMaxPos= np.zeros(iMaxPos,)
    piHistIntensity = GetIntensityHistogram256(img)
    # first smooth the distribution
    piHistSmooth = SmoothHistogram256(piHistIntensity, 1)
    # compute gradient
    piHistGrad = GradientOfHistogram256(piHistSmooth)

    # check for zeros
    iCntMaxima = 0;
    for i in range(iNumBins-2,2,-1):
        if iCntMaxima < iMaxPos:
            break
        if ((piHistGrad[i-1] < 0) and (piHistGrad[i] > 0)):
            iSumAroundMax = piHistSmooth[i-1] + piHistSmooth[i] + piHistSmooth[i+1];
            if (not (iSumAroundMax < iMaxPix1 and i < 64)):
                piMaxPos[iCntMaxima] = i
                iCntMaxima+=1

    iThresh = 0;

    if (iCntMaxima == 0):
        # no any maxima inside (only 0 and 255 which are not counted above)
        # Does image black-write already?
        iMaxPix2 = iMaxPix / 2;
        sum = 0
        for i in range(256): # select mean intensity
            sum += piHistIntensity[i];
            if (sum > iMaxPix2):
                iThresh = i;
                break;
    elif (iCntMaxima == 1):
        iThresh = piMaxPos[0]/2
    elif (iCntMaxima == 2):
        iThresh = (piMaxPos[0] + piMaxPos[1])/2;
    else: # iCntMaxima >= 3
        #CHECKING THRESHOLD FOR WHITE
        iIdxAccSum = 0
        iAccum = 0
        for i in range(iNumBins - 1,0,-1):#(int i = iNumBins - 1; i > 0; --i)
            iAccum += piHistIntensity[i];
            # iMaxPix/18 is about 5,5%, minimum required number of pixels required for white part of chessboard
            if ( iAccum > (iMaxPix/18) ):
                iIdxAccSum = i
                break

        iIdxBGMax = 0
        iBrightMax = piMaxPos[0]

        for n in range(iCntMaxima - 1): #(unsigned n = 0; n < iCntMaxima - 1; ++n)
            iIdxBGMax = n + 1;
            if ( piMaxPos[n] < iIdxAccSum ):
                break;
            iBrightMax = piMaxPos[n];

        # CHECKING THRESHOLD FOR BLACK
        iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];

        #IF TOO CLOSE TO 255, jump to next maximum
        if (piMaxPos[iIdxBGMax] >= 250 and iIdxBGMax + 1 < iCntMaxima):
            iIdxBGMax+=1
            iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]]


        for n in range(iIdxBGMax + 1, iCntMaxima): #(unsigned n = iIdxBGMax + 1; n < iCntMaxima; n++)
            if (piHistIntensity[piMaxPos[n]] >= iMaxVal):
                iMaxVal = piHistIntensity[piMaxPos[n]]
                iIdxBGMax = n


        #SETTING THRESHOLD FOR BINARIZATION
        iDist2 = (iBrightMax - piMaxPos[iIdxBGMax])/2
        iThresh = iBrightMax - iDist2
    

    if (iThresh > 0):
        img = (img >= iThresh);
    return img




def order_points_old(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")


def order_points(pts):
    center = np.mean(pts,axis=0)
    np2 = pts-center
    np2[:,1] = -np2[:,1]
#    angle = (np.arctan2(np2[:,0],np2[:,1])+2*np.pi) % (2*np.pi)
    angle = np.arctan2(np2[:,1],np2[:,0])
    idx = np.argsort(angle)
#    print(pts)
    pts=pts[idx[::-1],:]
#    print(pts)
    return pts

#arrange quads order based upon long_quad    
def arrangeQuads(quads,long_quad):
    ordered_quads=[]
    for i in range(len(quads)):
        pts = np.squeeze(quads[i])
        new_pts = order_points(pts)
        ordered_quads.append(new_pts)
        
    ref_v = long_quad[1,:]-long_quad[0,:]
    ref_v = ref_v/np.linalg.norm(ref_v)
    for i in range(len(ordered_quads)):
        q = ordered_quads[i]
        max_dis = 0
        cur_idx = -1
        for j in range(0,4):
            v = long_quad[(j+1)%4,:]-long_quad[j,:]
            v = v/np.linalg.norm(ref_v)
            cur_dis = np.dot(v,ref_v)
            if cur_dis>max_dis:
                max_dis = cur_dis
                cur_idx = j
        ordered_quads[i] = np.roll(q,-cur_idx,axis=0)
    return ordered_quads


def findPtNeighbors(pt, quads, corner_map, cur_grid_x,cur_grid_y, min_dis_thd,firstRun = True):

    if firstRun is False:
        corner_map[cur_grid_x,cur_grid_y,:2] = pt
#        print(corner_map[cur_grid_x,cur_grid_y,:2])
    min_dist = 1e10
    quad_idx =-1
    corner_idx =-1
    for i in range(len(quads)):
        for j in range(4):
            pt2 = quads[i][j,:]
            dis = np.linalg.norm(pt-pt2)
            if dis<min_dist:
                min_dist = dis
                quad_idx = i
                corner_idx = j
    
    if min_dist<20: #found the grid #fixed a bug 0408
        cur_quad = quads[quad_idx]
        if firstRun is False:
            corner_map[cur_grid_x,cur_grid_y,2:] = cur_quad[corner_idx,:]
#            print(corner_map[cur_grid_x,cur_grid_y,2:])
        quads.pop(quad_idx)
        gridxy=np.array([0,0])
        if corner_idx==1:
            gridxy=np.array([1,0])
        elif corner_idx==2:
            gridxy=np.array([1,1])
        elif corner_idx ==3:
            gridxy=np.array([0,1])
    
        
        for k in range(4):
            if k == corner_idx and firstRun is False:
                continue
            pt = cur_quad[k,:]
            if k==0:
                new_gridxy = np.array([0,0])
            elif k==1:
                new_gridxy = np.array([1,0])
            elif k==2:
                new_gridxy = np.array([1,1])
            elif k==3:
                new_gridxy = np.array([0,1])
            v = new_gridxy-gridxy
            new_grid_x = cur_grid_x +v[0]
            new_grid_y = cur_grid_y +v[1]
            if new_grid_x<0 or new_grid_x>=corner_map.shape[0] or \
                new_grid_y<0 or new_grid_y>=corner_map.shape[1]:
#                print([new_grid_x, new_grid_y])

                continue
            quads,corner_map = findPtNeighbors(pt, quads, corner_map, new_grid_x, new_grid_y, min_dis_thd, False)
    return quads,corner_map

# Find the angle and distance, distance represents the closest dis between points of two quads
def findAngleDisBetweenQuads(q1,q2):
    # q1 and q2 are both 4x2 array 
    maxDis = 1e6
    for i in range(4):
        for j in range(4):
            curDis = np.linalg.norm(q1[i,:]-q2[i,:])
            if curDis < maxDis:
                maxDis = curDis
    v11 = q1[0,:]-q1[1,:]
    v12 = q1[1,:]-q1[2,:]
    v21 = q2[0,:]-q2[1,:]
    v22 = q2[1,:]-q2[2,:]
    
    if np.linalg.norm(v11)>np.linalg.norm(v12):
        v1 = v11
    else:
        v1 = v12
    if np.linalg.norm(v21)>np.linalg.norm(v22):
        v2 = v21
    else:
        v2 = v22    
        
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)

    d = np.dot(u1, u2)
    angle = np.abs(np.arccos(d)*180/np.pi-90)
    if np.linalg.norm(v1)>np.linalg.norm(v2):
        curQ = q1
    else:
        curQ = q2
        
    return maxDis, angle, curQ

def findAngleOfQuad(q):
    # q is 4x2 array 
    epsilon = 1e-7

    v1 = q[0,:]-q[1,:]
    v2 = q[1,:]-q[2,:]
    v3 = q[3,:]-q[2,:]
    v4 = q[0,:]-q[3,:]

        
    u1 = v1 / (np.linalg.norm(v1)+epsilon)
    u2 = v2 / (np.linalg.norm(v2)+epsilon)
    u3 = v3 / (np.linalg.norm(v3)+epsilon)
    u4 = v4 / (np.linalg.norm(v4)+epsilon)

    d1 = np.dot(u1, u3)
    angle1 = np.abs(np.arccos(d1)*180/np.pi)
    d2 = np.dot(u2, u4)
    angle2 = np.abs(np.arccos(d2)*180/np.pi)
    d3 = np.dot(u1, u2)
    angle3 = np.abs(np.arccos(d3)*180/np.pi)
    d4 = np.dot(u3, u4)
    angle4 = np.abs(np.arccos(d4)*180/np.pi)

    return angle1, angle2,angle3,angle4
    
def findChessBoardNeighbors(quads, longquads):
    #assert(len(longquads)==1 or len(longquads)==2)
    total_area = 0
    for i in range(len(quads)):
        total_area +=  cv2.contourArea(quads[i], False)
    avg_dis = np.sqrt(total_area/len(quads))

    #arrange longquads to clockwise
    for i in range(len(longquads)):
        pts = np.squeeze(longquads[i])
        pts = order_points(pts)
        longquads[i] = pts
        
    if len(longquads)==2:
        q0 = longquads[0]
        v1 = q0[0,0:]-q0[1,:]
        v2 = q0[1,:]-q0[2,:]
        d0 = np.linalg.norm(v1)+np.linalg.norm(v2)
        q1 = longquads[1]
        v1 = q1[0,:]-q1[1,:]
        v2 = q1[1,:]-q1[2,:]
        d1 = np.linalg.norm(v1)+np.linalg.norm(v2)
        if d0>d1:
            cur_quad = longquads[0]
        else:
            cur_quad = longquads[1]
    elif len(longquads)==1:
        cur_quad = longquads[0]
    else:
        cur_quad = longquads[0]
        maxAngle = 1e6
        for i in range(len(longquads)):
            for j in range(len(longquads)):
                if i!= j:
                    q1 = longquads[i]
                    q2 = longquads[j]
                    curDis, cur_angle, curQ = findAngleDisBetweenQuads(q1,q2)
                    if curDis <avg_dis*2 and cur_angle <maxAngle:
                        maxAngle = cur_angle
                        cur_quad = curQ
        # find the two perpenticular quad
            # find long side 
        
    v1 = cur_quad[1,:]-cur_quad[0,:]
    v2 = cur_quad[2,:]-cur_quad[1,:]
    d3 = np.linalg.norm(v1)
    d4 = np.linalg.norm(v2)
    if d3>d4:
        v = v1
        if v[0]>0:
            cur_idx=0
        else:
            cur_idx=2
    else:
        v = v2
        if v[0]>0:
            cur_idx=1
        else:
            cur_idx=3
            
    cur_quad=np.roll(cur_quad,-cur_idx,axis=0)
    anchor = cur_quad[0,:]
    #clean quads
    new_quads = arrangeQuads(quads,cur_quad)
    #26x26 grid x,y and 2 pts possible for each corner, total 26x26x4
    corner_map = -np.ones((BOARDWIDTH,BOARDHEIGHT,4))
    
    cur_grid_x = 11 #these are hard coded because we know the location
    cur_grid_y = 12
    min_dist = 1e10
    quad_idx =-1
    corner_idx =-1
    for i in range(len(quads)):
        for j in range(4):
            pt2 = quads[i][j,:]
            dis = np.linalg.norm(anchor-pt2)
            if dis<min_dist:
                min_dist = dis
                quad_idx = i
                corner_idx = j
    if min_dist <max(avg_dis,min(d3,d4)*2):
        anchor = quads[quad_idx][corner_idx,:]
        new_quads,corner_map = findPtNeighbors(anchor, new_quads, corner_map, cur_grid_x, cur_grid_y, avg_dis)
    
    final_map = -np.ones((corner_map.shape[0],corner_map.shape[1],2))
    for i in range(corner_map.shape[0]):
        for j in range(corner_map.shape[1]):
            p1 = corner_map[i,j,:2]
            p2 = corner_map[i,j,2:]
            if p1[0]>0 and p2[0]>0:
                final_map[i,j,:] = (p1+p2)/2
            elif p1[0]>0:
                final_map[i,j,:] = p1
            elif p2[0]>0:
                final_map[i,j,:] = p2    
    return final_map, corner_map

def getROIfromImg(im,pt,winx,winy):
    px = int(round(pt[0]))
    py = int(round(pt[1]))
    lefttop=[px-winx,py-winy]
    if lefttop[0]<0:
        lefttop[0]=0
    if lefttop[1]<0:
        lefttop[1]=0    
    rightbottom=[px+winx,py+winy]
    if rightbottom[0]>im.shape[1]:
        rightbottom[0]=im.shape[1]
    if rightbottom[1]>im.shape[0]:
        rightbottom[1]=im.shape[0]    
    return lefttop, im[lefttop[1]:rightbottom[1],lefttop[0]:rightbottom[0]]
        
def detectCorner(im, frontDetect = False, minSize = None, allowTwoPointContour= False):
    try:
        blk_sz = min(im.shape[:2])//10
        if blk_sz%2==0:
            blk_sz+=1
        th2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blk_sz,3)
        gap=2
        th2[:gap,:]=0
        th2[-gap:,:]=0
        th2[:,:gap]=0
        th2[:,-gap:]=0

        # if frontDetect is False:
        #     subpix_sz = 11
        # else:
        #     subpix_sz = 21
        #for dialations in range(min_dilations,max_dilations+1):
        #    thresh_img_new = cv2.dilate( thresh_img_new,  kernel,iterations =1);
        if allowTwoPointContour is True:
            kernelSz = 5
        else:
            kernelSz = 7
        kernel = np.ones((kernelSz,kernelSz),np.uint8)
        th2 = cv2.dilate( th2,  kernel,iterations =1);
        quads,long_quads = generateQuads(th2, frontDetect,minSize,allowTwoPointContour)
        draw_quads = quads.copy()
        draw_longquads = long_quads.copy()
        final_map, org_map = findChessBoardNeighbors(quads, long_quads)
        harris_im =  cv2.cornerHarris(im,7,3,0.04)
        total_area = 0
        for i in range(len(quads)):
            total_area +=  cv2.contourArea(quads[i], False)
        avg_dis = np.sqrt(total_area/len(quads))
        subpix_sz = int(avg_dis*0.2)
        if subpix_sz%2 ==0:
            subpix_sz+=1
        for i in range(final_map.shape[0]):
            for j in range(final_map.shape[1]):
                pt = final_map[i,j,:]
                if pt[0]>=0 and pt[1]>=0:
                    pt2 = np.expand_dims(np.expand_dims(pt,axis=0),axis=0).astype(np.float32)
                    pt_sub =cv2.cornerSubPix(im, pt2, (subpix_sz, subpix_sz), (-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 40, 0.01))
                    # pt_sub = cv2.cornerSubPix(im, pt2, (11, 11), (-1, -1),
                    #                           (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 0.1))
                    pt2 = final_map[i,j,:]
                    if True:
                       lt,  sub_harris = getROIfromImg(harris_im,pt,subpix_sz,subpix_sz)
                       ret,thd_harris = cv2.threshold(sub_harris,0.001*sub_harris.max(),1,0)
                       if thd_harris[int((pt_sub[0,0,1]-lt[1])),int((pt_sub[0,0,0]-lt[0]))]==0: # means the detect point is off
                           ind = np.unravel_index(np.argmax(sub_harris, axis=None), sub_harris.shape)
                           newpt = [lt[0]+ind[1],lt[1]+ind[0]]
                           # pt_sub =cv2.cornerSubPix(im, newpt, (subpix_sz, subpix_sz), (-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 40, 0.001));
                           pt_sub = newpt
                    # pt = pt2
                    final_map[i,j,:] = np.squeeze(pt_sub)
        result = True
        if not len(quads) or not len(long_quads):
            result = False
        else:
            if len(quads)<20:
                result = False
    except:
        result = False
        draw_quads = []
        draw_longquads = []
        final_map= []
        org_map = []

            
    return result, draw_quads,draw_longquads, final_map, org_map
    

