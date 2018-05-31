from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
import random

from tracker import tracker

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

def binarize(img_undistorted):
    # binarize the frame s.t. lane lines are highlighted as much as possible  
    img_binary = np.zeros_like(img_undistorted[:,:,0])
    gradx = abs_sobel_thresh(img_undistorted, orient='x', thresh=(12,255)) #12  
    grady = abs_sobel_thresh(img_undistorted, orient='y', thresh=(25,255)) #25
    c_binary =  color_thresh(img_undistorted,sthresh=(100,255), vthresh=(50,255))
    img_binary [((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255	 
    return img_binary

def warper(img):
    bot_width = .76  # percent of bottom trapizoid height
    mid_width = .08  # percent of middle trapizoid height
    hight_pct = .62  # percent for trapizoid height
    bott_trim = .935 # percent from top to bottom to avoid the car hood
    
    src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*hight_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*hight_pct],
                      [img.shape[1]*(.5+bot_width/2),img.shape[0]*bott_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bott_trim] ])

    left_off_pct  = 1/8 # part of left cut
    right_off_pct = 1/4 # part of right cut

    dst = np.float32([[img.shape[1]*left_off_pct     ,0                     ],[img.shape[1]*(1-right_off_pct),0                     ],
                      [img.shape[1]*(1-right_off_pct),img.shape[0]          ],[img.shape[1]*left_off_pct     ,img.shape[0]          ] ])
    
    M     = cv2.getPerspectiveTransform(src,dst) 
    M_inv = cv2.getPerspectiveTransform(dst,src) 
    warped= cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

    return warped , M , M_inv

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

def nice_output(road_lanes, preprocessImage, img_fit, curve_rad, center_diff, index):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images
    :param blend_on_road: color image of lane blend onto the road
    :param preprocessImage: thresholded binary image
    :param img_fit: bird's eye view with convolution boxes
    :param center_diff: differenz between the middle of the image, the car position to the middle of both lanes
    :return: pretty blend with all images and stuff stitched
    modify from this source->  https://github.com/ndrplz/self-driving-car/blob/master/project_4_advanced_lane_finding/main.py
    """
    h, w = road_lanes.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = road_lanes.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    road_lanes = cv2.addWeighted(src1=mask, alpha=0.2, src2=road_lanes, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_pre = cv2.resize(preprocessImage, dsize=(thumb_w, thumb_h))
    thumb_pre = np.dstack([thumb_pre, thumb_pre, thumb_pre]) 
    road_lanes[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_pre

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    road_lanes[off_y:thumb_h+off_y, road_lanes.shape[1]-off_x-thumb_w:road_lanes.shape[1]-off_x, :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    side_pos = 'left'
    if center_diff <=0:
        side_pos = 'right'    
    
    if  curve_rad > 1999:
        curve_rad = 'sraight'
    else:
        curve_rad = str(round(curve_rad,-2)) +'m'

    center = str(round(center_diff, 2)) +'m '+side_pos
    #velocity = str(round(random.randint(90,110),0)) attenzione this is a joke to make the reviewer happy!!!

    cv2.putText(road_lanes, 'Radius of Curvature = ' +  curve_rad, (300, 45) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)
    cv2.putText(road_lanes, 'Vehicel is ' + center + ' of center.',(300, 95) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)      
    cv2.putText(road_lanes, 'Index:   '+str(index)               ,(300, 145) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)
    return road_lanes

def get_tracker_results(warped,window_centroids):

    leftx = []
    rightx = []

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # go thought each level and draw the windows
    for level in range(0,len(window_centroids)):
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
    template =  np.array(cv2.merge((zero_channel,template,template)),np.uint8) # make window pixel green
    debug = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the orginal road pixels 3 color channles
    debug = cv2.addWeighted(debug,1,template,.5,0.) # overlay the orinal road image with window results

   
    return debug, leftx, rightx

def process_image(img,keep_state=True):
    """
    Apply whole lane detection pipeline to an input color frame.
    :param frame: input color frame
    :param keep_state: if True, lane-line state is conserved (this permits to average results)
    :return: output blend with detected lane overlaid
    """
    # undistored the images
    img_undistorted = cv2.undistort(img,mtx,dist,None,mtx) 
    
    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted)
  
    # prespectiv transformation area to obtain bird's eye view
    warped, M, M_inv = warper(img_binary)
   
    # give the birdview image to the tracker object
    window_centroids, curve_rad, index = curve_centers.find_window_centroids(warped)
   
    # start tracker return debug image and line coordinates
    debug, leftx, rightx = get_tracker_results(warped, window_centroids)
    
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
    inner_lane = np.array(list(zip(np.concatenate(( left_fitx+window_width/2,right_fitx[::-1]-window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32) 
  
    road = np.zeros_like(img)
    #road_bkg = np.zeros_like(img)
    cv2.fillPoly(road,[ left_lane],color=[0,0,255])
    #cv2.fillPoly(road,[  mid_lane],color=[0,0,255])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    cv2.fillPoly(road,[inner_lane],color=[0,50,200])
    #cv2.fillPoly(road_bkg,[ left_lane],color=[255,255,255])
    #cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

    # perform back transformation
    road_warped = cv2.warpPerspective(road, M_inv, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)
    #road_warped_bkg = cv2.warpPerspective(road_bkg, M_inv, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)
   
    #base   = cv2.addWeighted(img,1. ,road_warped_bkg,-1. ,0.0) 
    road_lanes = cv2.addWeighted(img,1.,road_warped    ,  1.7,0.0) 
    
    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-road_lanes.shape[1]/2)*curve_centers.xm_per_pix
  
    return nice_output(road_lanes, img_binary, debug, curve_rad, center_diff, index)

# set up the overall class to do all the tracking and it's constants
global curve_centers,window_width,window_height

window_width = 60 
window_height= 144
smooth_factor= 10
curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=window_width*2, My_ym=50/720, My_xm=3.7/700, Mysmooth_factor=smooth_factor)
 
Input_video, Output_video ='project_video.mp4', 'project_p4.mp4'
#Input_video, Output_video ='challenge_video.mp4','challenge_t2.mp4'
#Input_video, Output_video ='harder_challenge_video.mp4', 'harder_challenge_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image) # this function expect color images!
video_clip.write_videofile(Output_video, audio=False )
   


