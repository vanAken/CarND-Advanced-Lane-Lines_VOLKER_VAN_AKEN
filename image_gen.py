import numpy as np
import cv2
import pickle
import glob
from tracker import tracker
from sklearn import linear_model, datasets

#import matplotlib.pyplot as plt

# Read the camera calibration result 
dist_pickle = pickle.load(open( "./camera_cal/camera_cal.p", "rb" ) )
mtx = dist_pickle["mtx"] 
dist = dist_pickle["dist"]

def dir_s_threshold(image, sobel_kernel=15, thresh=(0.7, 1.2)):
    # s-CHANEL from i-net
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(hls[:,:,2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(hls[:,:,2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Return the binary image
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

# Define a function to return the magnitude of the gradient
#  from i-net
def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray[:,:,2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray[:,:,2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def abs_sobel_thresh(img, orient="x", sobel_kernel=3, thresh=(0,255)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # process x or y
    if orient == "x":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == "y":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    scaled_sobel = np.uint8(255* abs_sobel/ np.max(abs_sobel))
    # Return the binary image
    binary_output =  np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2+sobely**2)
    
    scaled_gradmag = np.uint8(255* gradmag/np.max(gradmag))
    # Return the binary image
    binary_output =  np.zeros_like(scaled_gradmag)
    binary_output[(scaled_gradmag >= thresh[0]) & (scaled_gradmag <= thresh[1])] = 1
    return binary_output

def dir_thresh(image, sobel_kernel=15, thresh=(0,np.pi/2)):
    # GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2gray)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Return the binary image
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_thresh(img, sthresh=(0, 255), vthresh=(0, 255)):
    # S-CHANNEL
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary_output =  np.zeros_like(s_channel)
    s_binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    # V-CHANNEL
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary_output =  np.zeros_like(v_channel)
    v_binary_output[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    # Combine the two binary thresholds 
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(s_binary_output == 1) & (v_binary_output == 1)] = 1
    return combined_binary 
    return binary_output

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

# Make a list of calibration images
images = glob.glob('./test_images/test*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    # read images
    img = cv2.imread(fname)
    # undistored the images
    img = cv2.undistort(img,mtx,dist,None,mtx) 
   
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255)) #12  
    grady = abs_sobel_thresh(img, orient='y', thresh=(25,255)) #25
    c_binary =  color_thresh(img,sthresh=(100,255), vthresh=(50,255))
    preprocessImage [((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255	 
  
    # prespectiv transformation area
    bot_width = .76  # percent of bottom trapizoid height
    mid_width = .08  # percent of middle trapizoid height
    hight_pct = .62  # percent for trapizoid height
    bott_trim = .935 # percent from top to bottom to avoid the car hood
    
    src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*hight_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*hight_pct],
                      [img.shape[1]*(.5+bot_width/2),img.shape[0]*bott_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bott_trim] ])
    left_off_pct  = 1/8
    right_off_pct = 1/4
    dst = np.float32([[img.shape[1]*left_off_pct     ,0                     ],[img.shape[1]*(1-right_off_pct),0                     ],
                      [img.shape[1]*(1-right_off_pct),img.shape[0]          ],[img.shape[1]*left_off_pct     ,img.shape[0]          ] ])
    # perform transformation
    M     = cv2.getPerspectiveTransform(src,dst) 
    M_inv = cv2.getPerspectiveTransform(dst,src) 
    warped= cv2.warpPerspective(preprocessImage, M, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)
    
    window_width = 60
    window_height= 72
    
    # set up the overall class to do all the tracking
    curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=120, My_ym=100/720, My_xm=4/384, Mysmooth_factor=15)
    
    window_centroids = curve_centers.find_window_centroids(warped)
    
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # points used to find the left and the right lanes and the middle
    leftx  = []
    rightx = []

    # go thought each level and draw the windows
    for level in range(0,len(window_centroids)):
        # Window_mask is a funktion to draw areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # add center value found in frame to the list of lane points per left, right
        leftx.append (window_centroids[level][0])  
        rightx.append(window_centroids[level][1])
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8)# add both left and right window pixels togehter
    zero_channel = np.zeros_like(template) # create a zero color channle
    template =  np.array(cv2.merge((zero_channel,zero_channel,template)),np.uint8) # make window pixel green
    debug = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the orginal road pixels 3 color channles
    debug = cv2.addWeighted(debug,1,template,.5,0.) # overlay the orinal road image with window results

    # fit the lane boundaries to the left, right center positions found
    yvals = range(0,warped.shape[0])

    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

    left_fit  = np.polyfit(res_yvals, leftx,2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals+ left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)

    right_fit  = np.polyfit(res_yvals, rightx,2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals+ right_fit[2]
    right_fitx = np.array(right_fitx,np.int32)

    left_lane  = np.array(list(zip(np.concatenate(( left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32) 
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32) 
    mid_marker = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32) 

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)

    cv2.fillPoly(road,[ left_lane],color=[255,0,0])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    #cv2.fillPoly(road_bkg,[ left_lane],color=[255,255,255])
    #cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

    # perform back transformation
    road_warped = cv2.warpPerspective(road, M_inv, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)
    #road_warped_bkg = cv2.warpPerspective(road_bkg, M_inv, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)
   
    #base   = cv2.addWeighted(img,1. ,road_warped_bkg,-1.,0.) 
    result = cv2.addWeighted(img,.8 ,road_warped,1.,0.) 
    
    ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dim 
    xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dim

    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix,2)
    curve_rad = ((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <=0:
        side_pos = 'right'

    #draw
    cv2.putText(result,'Radius of Curvature = '+str(round(curve_rad,3))+'m',(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result,'vehicel is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    write_name = './output_images/debug'+fname.split('/')[-1]
    cv2.imwrite (write_name,debug)

    write_name = './output_images/pre2'+fname.split('/')[-1]
    cv2.imwrite (write_name,preprocessImage)

    write_name = './output_images/result'+fname.split('/')[-1]
    cv2.imwrite (write_name,result)




