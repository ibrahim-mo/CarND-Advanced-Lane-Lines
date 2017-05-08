## Writeup for the Advanced Lane Finding Project

### Ibrahim Almohandes, 5/7/2017

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calib_undist1.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_1.png "Binary Example"
[image4]: ./output_images/undist2warped_1.png "Warp Example"
[image5]: ./output_images/marked_lines_1.png "Fit Visual"
[image6]: ./output_images/frame_913.40_961.69.jpg "Output"
[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 9 through 54 of the file called `calib_cam.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the test images using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function `color_grad()` at lines 79 through 109 in `proj4.py`).  In this function, I first converted a copy of the image into HLS color space, then I applied x-gradient thresholding on the L channel, and color thresholding on the S channel. Finally, I combined the two binary images to create the thresholded binary result. Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 72 through 77 in the file `proj4.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner (lines 344-347 in `proj4.py`):

```python
offset = 290
src = np.float32([[580,460],[700,460],[1100,img_size[1]],[220,img_size[1]]])
dst = np.float32([[offset,0],[img_size[0]-offset,0],
                [img_size[0]-offset,img_size[1]],[offset,img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 290, 0        | 
| 700, 460      | 990, 0        |
| 1100, 720     | 990, 720      |
| 220, 720      | 290, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I wrote two functions to identify lane line pixels and fit them with 2nd degree polynomials, as follows:

The first function is `find_lines1()` at lines 111-201 of `proj4.py`. This function detects the pixels of lane lines from scratch, then use them to create the left and right lines' 2nd degree polynomials. This function is suitable for single images, and for initial frames of video clips, or when the previous frame(s) is/are lost or not good and we need to recalculate line pixels using the whole frame (image).

The second function is `find_lines2()` at lines 204-262 of `proj4.py`. This function saves time by starting from the fitted lines of most recent good frame, then it searches only in an area around these lines (with a specified margin) to find the most likely pixels of the new lines, and finally use them to create the new 2nd degree fitted lines. 

I select one of the two functions at lines 357-362 in `proj4.py` (function `image_pipeline()`) to fit lane lines pixels with 2nd order polynomials like the ones in this output (Each of the two lines has its own polynomial coefficients):

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in a function called `measure_curve_pos()` at lines 265 through 302 in my code in `proj4.py`. I start by taking the bottom pixels of the two fitted lane lines, and with the previously calculated 2nd-order polynomial coefficients, I calculate the curvatures according to the equation in these code lines:

```python
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])  
```

Then I convert the curvatures to meters according to these ratios:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

Finally, I calculate the offset of the car center with respect to the center of the lane (also warped image) and convert them from pixels to meters with the same ratios.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Final steps include unwarping the binary image (function `unwarper()` at lines 304-323), then combining it with the original image (lines 375-378 in `proj4.py`). Top-level function `image_pipeline()` at lines 325-411 in `proj4.py` calls all the sub-functions implementing the pipeline steps. Finally `image_pipeline()` gets called to create the output image (or frame of a video).  Here is an example of my result on a test image (frame form the output video):

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

After doing camera calibration (once), I broke the pipeline's functionality into the following functions:

`undistort()`, `warper()`, `color_grad()`, `find_lines1()`, `find_lines2()`, `measure_curve_pos()`, and `unwarper()`. Then, I created a top-level function called `image_pipleline()` which calls all these sub-functions in a controlled manner, and finally returning the output image (or frame). `image_pipeline()` can be called directly (for images) or passed as a callback function to `VideoFileClip.fl_image()` which allows it to process one frame at a time.

Here's a brief of some of the challenges I faced during this project (esp. when processing the project video):

Some video frames produced very bad lane lines, such as constituting very wide or very narrow lanes, or having too small radius of curvature (very curvy lines), or discrepancy between the two lane lines (not semi-parallel, or one almost flat while the other very bent). To solve these issues, I checked for such conditions in my `image_pipleline()` and excluded these frames from creating fitted lines, while generally smoothing out the lines by keeping track of the last `n=10` good frames' polynomials and averaging their coefficients to creating a smooth fitted line at each frame.

Another technique I used to overcome varying lighting conditions in the project video was to use the HLS color space (instead of RGB). Using the HLS to create the gradients (esp. S channel for color thresholding) along with an appropriate thresholding range produces better results under bright light and in shady areas.

Although my model does reasonably good with the basic project video (albeit with some wobbling around bright areas), it still needs more improvement to handle more difficult situations, like dark lines that can get mistaken by the model for being lane lines (as in the challenge video), or more severe variation in lighting conditions. Challenges like these can be pursued further beyond this project.

Thanks for reading!
