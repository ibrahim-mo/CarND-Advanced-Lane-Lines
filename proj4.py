import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a class to receive the characteristics of each line detection
class Line():
    n = 10 #class variable
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        #x values of the last n fits of the line
        #self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     #x values corresponding to best_fit
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = None #[np.array([False])]
        #polynomial coefficients of the last n fits of the line
        self.recent_fits = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update_line(self):
        if len(self.recent_fits) == Line.n:
            self.recent_fits.pop(0)
        self.recent_fits.append(self.current_fit)
        recent_fits_arr = np.array(self.recent_fits)
        self.best_fit = np.mean(recent_fits_arr, axis=0)
        ploty = self.ally
        self.bestx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]

line1 = Line()
line2 = Line()

def plot_results(image1, image2, title1, title2, gray1=False, gray2=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    #f.tight_layout()
    if not gray1:
        ax1.imshow(image1)
    else:
        ax1.imshow(image1, cmap='gray')
    ax1.set_title(title1, fontsize=20)
    if not gray2:
        ax2.imshow(image2)
    else:
        ax2.imshow(image2, cmap='gray')
    ax2.set_title(title2, fontsize=20)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def undistort(img):
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "camera_cal/mtx_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undist = cv2.undistort(img, mtx, dist)
    return undist

def warper(img, src, dst):
    img_size = img.shape[1::-1] #(img.shape[1], img.shape[0])
    # Compute and apply perpective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def color_grad(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V/S channel
    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    # v_channel = hsv[:,:,2]
    # s_channel = hsv[:,:,1]
    # Convert to HLS color space and separate the L/S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    # sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    #return color_binary
    comb_binary = np.zeros_like(s_channel).astype(np.uint8)
    comb_binary[(sxbinary==1) | (s_binary==1)] = 1
    return comb_binary

def find_lines1(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    #plt.show()

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    line1.current_fit, line2.current_fit = left_fit, right_fit
    line1.allx, line2.allx = left_fitx, right_fitx
    line1.ally = line2.ally = ploty
    line1.detected = line2.detected = True
    #return left_fit, right_fit, left_fitx, right_fitx, ploty

def find_lines2(binary_warped):
    left_fit, right_fit = line1.current_fit, line2.current_fit
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    line1.current_fit, line2.current_fit = left_fit, right_fit
    line1.allx, line2.allx = left_fitx, right_fitx
    line1.ally = line2.ally = ploty
    #return left_fit, right_fit, left_fitx, right_fitx, ploty

def measure_curve_pos():
    left_fit, right_fit = line1.current_fit, line2.current_fit
    left_fitx, right_fitx = line1.allx, line2.allx
    ploty = line1.ally
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    # lane width
    lane_width = (right_fitx[-1] - left_fitx[-1]) * xm_per_pix
    #print('Current lane width = {:.2f}m'.format(lane_width))

    # position of the vehicle
    mid_pix = (left_fitx[-1] + right_fitx[-1])//2
    offset_x = (640 - mid_pix) * xm_per_pix

    line1.line_base_pos, line2.line_base_pos = (640 - left_fitx[-1]) * xm_per_pix, (right_fitx[-1] - 640) * xm_per_pix
    line1.radius_of_curvature, line2.radius_of_curvature = left_curverad, right_curverad

    return offset_x, lane_width

def unwarper(warped, src, dst):
    #left_fitx, right_fitx = line1.allx, line2.allx
    left_fitx, right_fitx = line1.bestx, line2.bestx
    ploty = line1.ally
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped) #.astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    return newwarp

def image_pipeline(img):
    """
    if not img.any():
        line1.detected = line2.detected = False
        return None
    """

    # Undistort the image using calibrated parameters mtx & dist
    undist = undistort(img)
    # Plot the result
    # plot_results(img, undist, 'Original Image', 'Undistorted Result')

    # Apply x-gradient and color thresholding on the L & S chnnels resp.
    binary_undist = color_grad(undist)
    # Plot the result
    # plot_results(undist, binary_undist, 'Undistorted Image', 'Thresholded Binary Result', gray2=True)

    # Warp the undistorted image
    img_size = img.shape[1::-1] #(img.shape[1], img.shape[0])
    offset = 290
    src = np.float32([[580,460],[700,460],[1100,img_size[1]],[220,img_size[1]]])
    dst = np.float32([[offset,0],[img_size[0]-offset,0],[img_size[0]-offset,img_size[1]],[offset,img_size[1]]])
    binary_warped = warper(binary_undist, src, dst)
    # Plot the result
    # undist_copy = np.dstack((binary_undist, binary_undist, binary_undist))*255
    # cv2.polylines(undist_copy, np.int32([src]), True, (250,0,0), 5)
    # warped_copy = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # cv2.polylines(warped_copy, np.int32([dst]), True, (250,0,0), 5)
    # plot_results(undist_copy, warped_copy, 
    #     'Undistorted Binary with src. pts. drawn', 'Warped Result with dst. pts. drawn')

    if not line1.detected or not line2.detected:
        # find initial lane lines
        find_lines1(binary_warped)
    else:
        # find lane lines based on previous frame
        find_lines2(binary_warped)

    # Measure curvature of the lane lines and position of the vehicle
    offset_x, lane_width = measure_curve_pos()

    if line1.radius_of_curvature > 200 and line2.radius_of_curvature > 200 and \
       line1.radius_of_curvature / line2.radius_of_curvature > 0.1 and \
       line1.radius_of_curvature / line2.radius_of_curvature < 10 and \
       lane_width >= 3.4 and lane_width <= 4:
        line1.update_line()
        line2.update_line()

    # warp back to the original
    newwarp = unwarper(binary_warped, src, dst)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    smaller_curverad = min(line1.radius_of_curvature, line2.radius_of_curvature)
    curv_str = 'Radius of curvature = {:.2f}(m)'.format(smaller_curverad)

    if offset_x > 0:
        offset_str = 'Vehicle is {:.2f}m left of center'.format(offset_x)
    elif offset_x < 0:
        offset_str = 'Vehicle is {:.2f}m right of center'.format(-offset_x)
    else:
        offset_str = 'Vehicle is at the center'

    #lane_str = 'Lane width ={:.2}'.format(lane_width)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,curv_str,(20,60),font,2,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result,offset_str,(20,120),font,2,(255,255,255),1,cv2.LINE_AA)
    #cv2.putText(result,lane_str,(20,180),font,2,(255,255,255),1,cv2.LINE_AA)

    #print(offset_str)
    #print(curv_str)
    """
    if line1.radius_of_curvature <= 200 or line2.radius_of_curvature <= 200 or \
        line1.radius_of_curvature / line2.radius_of_curvature <= 0.1 or \
        line1.radius_of_curvature / line2.radius_of_curvature >= 10 or \
        lane_width < 3.4 or lane_width > 4:
        #print('Warning: left and/or right line is too curved!')
        cv2.imwrite('tmp/frame_{:.2f}_{:.2f}.jpg'.format(line1.radius_of_curvature, line2.radius_of_curvature),
                     cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    """
    # plt.imshow(result)
    # plt.show()

    return result

# image = mpimg.imread('test_images/straight_lines1.jpg')
# print(image.shape) #(720, 1280, 3)
# image_pipeline(image)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

video_output = 'output_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
out_clip = clip1.fl_image(image_pipeline) #NOTE: this function expects color images!!
out_clip.write_videofile(video_output, audio=False)
