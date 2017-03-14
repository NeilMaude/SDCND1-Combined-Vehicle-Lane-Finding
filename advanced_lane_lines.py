# Project submission - advanced lane finding
# Neil Maude
# February 2017

# imports
import numpy as np
import cv2
import glob  # used for reading of files matching a pattern
import pickle
import os
import matplotlib.pyplot as plt

# Constants used in project
SOBEL_KERNAL_SIZE = 3  # default Sobel kernal size
SOBEL_ABS_THRESHOLDS = (20, 100)  # default Sobel absolute value thresholds
S_CHANNEL_THRESHOLDS = (90, 255)  # default S-channel thresholds

SOURCE_AREA = np.float32([[235, 700], [580, 460], [700, 460], [1070, 700]])  # default source for warp functions
DEST_AREA = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])  # default destination for warp functions

NUM_SLIDING_WINDOWS = 9  # default number of sliding windows to use when locating lines
MARGIN_SLIDING_WINDOWS = 100  # width of the windows +/- margin
MINIMUM_PIXELS_SLIDING = 50  # minimum number of pixels found to recenter window

OUTPUT_POLY_BACKGROUND_COLOUR = (0, 255, 0)         # colour for the polygon background
OUTPUT_LEFT_COLOUR = (0, 0, 255)
OUTPUT_RIGHT_COLOUR = (255,0,0)
OUTPUT_LINE_WEIGHT = 20

OVERLAY_WEIGHTING = 0.3

YM_PER_PIX = 30/720 # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meters per pixel in x dimension

MAX_RADIUS_DIFFERENCE = 0.25        # radii must be with in 25% of each other

# Utility functions
def create_dir(sDir):
    # Check if a directory exists, create it if it doesn't already exist
    d = os.path.dirname(sDir + '/')
    if not os.path.exists(d):
        os.mkdir(d)


def show_image(img, s_title='Image'):
    # Show a cv2 image in a window
    b, g, r = cv2.split(img)
    frame_rgb = cv2.merge((r, g, b))
    plt.imshow(frame_rgb)
    plt.title(s_title)
    plt.show()


# Part 1 - camera calibration
# Use a set of sample images for the camera to calibrate the camera and remove distortion
def calibrate_camera(samples_dir, sample_images_pattern, x_squares=9, y_squares=6, fVerbose=False,
                     verbose_dir='verbose_output'):
    # calibrate, using sample images matching a sample_images_pattern pattern in samples_dir
    # E.g. calibrate_camera('samples', 'sample*.jpg')
    # x_squares, y_squares used to set the chessboard size
    # fVerbose used to turn on/off verbose output, creation of example images etc

    if fVerbose:
        create_dir(verbose_dir)

    # Prepare object points, (0,0,0), (1,0,0), (2,0,0),..., (x_squares,y_squares,0)
    objp = np.zeros((x_squares * y_squares, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x_squares, 0:y_squares].T.reshape(-1, 2)

    # if fVerbose:
    #    print('Object points, objp: ', objp)  # I want to see what this looks like...

    # Arrays to store the obect points and image points from all the images
    objpoints = []  # 3d points in real world spacec
    imgpoints = []  # 2d points in the image plane

    # Get the list of calibration images
    images = glob.glob(samples_dir + '/' + sample_images_pattern)

    # Step over this list of images, searching for the chessboard corners
    x_size, y_size, x_new, y_new = 0, 0, 0, 0  # will find the width/height of the images and save this

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)  # Note: will be in BGR form
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Hence correct param for conversion to grayscale

        # extract and store the x,y size of the images and warn if not the same each time
        # will need this info later for the actual calibration
        x_new = img.shape[1]
        y_new = img.shape[0]
        if x_size > 0:
            if (x_new != x_size) or (y_new != y_size):
                # should have all of the input sizes the same
                print('Warning, images not the same size! ', idx)
        else:
            # save size of first image
            x_size = x_new
            y_size = y_new

        # Now find the corners
        ret, corners = cv2.findChessboardCorners(gray, (x_squares, y_squares), None)

        if ret == True:
            # Found some corners
            objpoints.append(objp)
            imgpoints.append(corners)  # Append the list of 3d corner points found by cv2

            # At this point, can output some images
            if fVerbose:
                print('Output chessboard corners image #', str(idx))
                cv2.drawChessboardCorners(img, (x_squares, y_squares), corners, ret)
                output_img_name = verbose_dir + '/corners' + str(idx) + '.jpg'
                cv2.imwrite(output_img_name, img)

    # got this far, have the object and image points
    if fVerbose:
        print('Found corners complete...')
        # print('Object points array: ', objpoints)
        # print('Image points array : ', imgpoints)

    # Now want to use these points to calibrate the camera
    img_size = (x_new, y_new)  # values saved earlier in the corners loop

    # Call the calibration function, using the object points and image points collected so far
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Now have all of the calibration matrices

    # Save the result for later, even though we're returning them from this function also
    cal_pickle = {}
    cal_pickle['mtx'] = mtx
    cal_pickle['dist'] = dist
    pickle.dump(cal_pickle, open('calibrate_pickle.p', 'wb'))

    # If we are going verbose this run, then undistort all of the sample images
    if fVerbose:
        images = glob.glob(samples_dir + '/' + sample_images_pattern)
        for idx, fname in enumerate(images):
            print('Output undistort image #', str(idx))
            img = cv2.imread(fname)
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            output_img_name = verbose_dir + '/undistort' + str(idx) + '.jpg'
            cv2.imwrite(output_img_name, dst)

    return mtx, dist  # return the calibration data


def load_calibration_data(pickle_file='calibrate_pickle.p'):
    # Helper function to re-load the calibration data from file
    cal_pickle = pickle.load(open(pickle_file, 'rb'))
    mtx = cal_pickle['mtx']
    dist = cal_pickle['dist']
    return mtx, dist


# Part 2 - undistort of a single image, using the camera calibration found in Part 1
# The main function here is going to be used as part of pipeline process later
# So the inputs are the image object (already read into memory) and the camera calibration matrices
def undistort_image(img, mtx, dist):
    # Undistort img, using the mtx/dist matrices
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# Part 3 - threshold detection
def mask2binary(mask):
    # convert an output mask from one of the thresholding functions into a binary image
    im_bw = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)[1]
    return im_bw


def sobel_abs_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Sobel - absolute thresholding in x or y direction
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Assumes cv2 was used to read the image and it's in BGR
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sbinary


def sobel_mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Sobel gradient magnitude thresholding
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Assumes cv2 was used to read the image and it's in BGR
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    gradmag = np.uint8(255 * gradmag / np.max(gradmag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def sobel_dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Sobel directional thresholding
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Assumes cv2 was used to read the image and it's in BGR
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    arctans = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(arctans)
    binary_output[(arctans >= thresh[0]) & (arctans <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def hls_s_channel(img, thresh=(0, 255)):
    # 1) Convert to HLS color space - could use any channel, but will use the S channel in this function
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)  # Assumes cv2 was used to read the image and it's in BGR
    # A reminder in case the other channels are ever of interest
    # H = hls[:, :, 0]
    # L = hls[:, :, 1]
    # S-channel is the channel required for this function
    S = hls[:, :, 2]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def combine_mask(a_mask, b_mask):
    # combine the two masks
    # assume same sizing
    combined_mask = np.zeros_like(a_mask)
    combined_mask[(a_mask == 1) | (b_mask == 1)] = 1
    return combined_mask


def preferred_threshold_image(img):
    # This function returns the preferred threshold process for the pipeline
    # Will use the hard coded values in the constants defined in this module and use some pre-selected set of filters
    s_binary = hls_s_channel(img, S_CHANNEL_THRESHOLDS)
    g_binary = sobel_abs_thresh(img, orient='x', sobel_kernel=SOBEL_KERNAL_SIZE, thresh=SOBEL_ABS_THRESHOLDS)
    combined_binary = combine_mask(s_binary, g_binary)
    img_binary = mask2binary(combined_binary)
    return img_binary

def preferred_threshold_image_from_file(image_file_name):
    img = cv2.imread(image_file_name)
    img_binary = preferred_threshold_image(img)
    return img_binary

# Part 4 - perspective transformation
def warp_image(img, src, dst):
    # warp an image using the src and dst boxes
    # expects an undistorted image as an input
    # src and dst should be arrays of points defining the source and destination zones
    # Example: src = np.float32([[235, 700], [580, 460], [700, 460], [1070, 700]])
    # Example: dst = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


def warp_original_to_overhead(img, src=SOURCE_AREA, dst=DEST_AREA):
    # takes an input image (usually a binary mask) and warps it using the src and dest params
    warped_image = warp_image(img, src, dst)
    return warped_image

def warp_overhead_to_original(img, src=SOURCE_AREA, dst=DEST_AREA):
    # takes a warped image and warps back to the original perspective
    # can do this by simply reversing the src/dest in a call to warp_image
    unwarped_image = warp_image(img, dst ,src)
    return unwarped_image

# Part 5 - lane finding
def find_lanes_sliding(binary_warped, nwindows=NUM_SLIDING_WINDOWS, margin=MARGIN_SLIDING_WINDOWS,
                       minpix=MINIMUM_PIXELS_SLIDING):
    # takes a warped binary input image and finds the lane lines
    # returns the polynomial fit parameters and pixels found
    # also returns a visualisation image of the fit - useful in debugging, but won't be used in pipeline

    # This is complex stuff - mainly provied in the lectures

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)      # added int cast to fix error when using from pipeline
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
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

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, leftx, lefty, rightx, righty, out_img


def find_lanes_search(binary_warped, left_fit, right_fit, margin=MARGIN_SLIDING_WINDOWS):
    # find lane lanes based on searching from an existing set of left/right fit lines
    # return the polynomial fit and pixels found
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    new_left_fit = np.polyfit(lefty, leftx, 2)
    new_right_fit = np.polyfit(righty, rightx, 2)

    return new_left_fit, new_right_fit, leftx, lefty, rightx, righty

def plot_lanes_on_warped(binary, left_fit, right_fit, overlay_weight=1):
    # takes a binary warped image and the left/right polynominals
    # draws the zone and lines on the image
    # returns the warped image with the overlay of the line

    # Generate some points for the edges
    binary_warped = np.zeros_like(binary)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Create a new image with the polygon
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((left_line, right_line))
    cv2.fillPoly(window_img, np.int_([line_pts]), OUTPUT_POLY_BACKGROUND_COLOUR)
    cv2.polylines(window_img, np.int_([left_line]), False, OUTPUT_LEFT_COLOUR, OUTPUT_LINE_WEIGHT)
    cv2.polylines(window_img, np.int_([right_line]), False, OUTPUT_RIGHT_COLOUR, OUTPUT_LINE_WEIGHT)
    # Merge the images to create the result
    result = cv2.addWeighted(out_img, 1, window_img, overlay_weight, 0)

    return result

def merge_over_camera_view(original_img, overlay_img):
    # merge a polygon showing detected lanes over the original camera image
    final_img = cv2.addWeighted(original_img, 1, overlay_img, OVERLAY_WEIGHTING, 0)
    return final_img

# Part 6 - Finding the lane curvature

# Calc the curvature
def get_curvature(leftx, lefty, rightx, righty):
    # takes in the pixels used to calculate the fit in the frame (need to fit again after scaling for meters)
    # returns the curvature (averaged over the two lanes), in meters

    y_eval = np.max(lefty)      # take the curvatures at the bottom of the image (max y value)

    # fit curves, applying the meters per pixel adjustments
    left_fit_cr = np.polyfit(lefty*YM_PER_PIX, leftx*XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(righty*YM_PER_PIX, rightx*XM_PER_PIX, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return (left_curverad + right_curverad) / 2.0, left_curverad, right_curverad

# Calc the offset from the centre of the lane, in meters
def get_offset(image_width, left_fit, right_fit, leftx, lefty, rightx, righty):
    # get the offset from the centre of the lane
    # will be +ve for car to the right of centre, -ve if to the left
    l_max = np.max(lefty)
    r_max = np.max(righty)
    if l_max > r_max:
        y_eval = l_max
    else:
        y_eval = r_max
    left_base = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_base = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_centre = (left_base + right_base) / 2.0
    car_centre = image_width /2
    offset = (car_centre - lane_centre) * XM_PER_PIX
    return offset

def add_text(img, radius, offset):
    # paste text over the image
    # return the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Radius = ' + str("{0:.2f}".format(radius)) + 'm', (10, 100), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, 'Offset = ' + str("{0:.2f}".format(offset)) + 'm', (10, 200), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
    return img

def check_similar_radii(r1, r2):
    # are the radii of the two curves similar?
    # if so, can use this fit
    max_diff = abs(MAX_RADIUS_DIFFERENCE * max(r1,r2))
    if abs(r1-r2) < max_diff:
        return True
    else:
        return False