# Run the image processing pipeline on video

import advanced_lane_lines as AdvLanes                          # main project code
import vehicle_detection as vd                                  # main project code
import numpy as np
import cv2

# Video processing tools
import imageio.plugins
imageio.plugins.ffmpeg.download()                               # get the MPEG codec if not already present
from moviepy.editor import VideoFileClip

# Params for lane detection
# Params
UPDATE_TEXT_INTERVAL = 10                                       # number of frames to average radius/offset
n_count_since_update = UPDATE_TEXT_INTERVAL                     # init, so we write to first frame
last_drawn_radius = 0                                           # last value, won't update every frame
last_drawn_offset = 0                                           # last value, won't update every frame
radius_rolling_count = 0                                        # rolling count for average calculation
offset_rolling_count = 0                                        # rolling count for average calculation
fFirstFrame = True                                              # are we on the first frame of the video?
fGoodLastFrame = False                                          # was the last frame good?
FRAMES_TO_KEEP = 5                                              # number of frames to retain for smoothing
SMOOTHING_WEIGHTS = [10,5,3,2,1]                                # weighting to apply to previous frame values
                                                                # when can't find the current frame

class Line():                                                   # class object to hold line data
    def __init__(self):
        self.fit = [0,0,0]
        self.points_x = []
        self.points_y = []

left_line = []                                                  # will retain a global list of the last few fits
right_line = []                                                 # will retain a global list of the last few fits

mtx, dist = AdvLanes.load_calibration_data()                    # load in the calibration data for the camera

# Params for vehicle detection
# Classifier settings - determined after searching parameter space
color_space_best = 'YCrCb' #'HSV'
spatial_size_best = (32,32)
hist_bins_best = 64
orient_best = 9
pix_per_cell_best = 8
cell_per_block_best = 2
hog_channel_best = 'ALL'

FRAMES_TO_KEEP_VEHICLE = 10                              # number of frames to retain for heatmap creation
FRAME_WEIGHTS = [1,1,1,1,1,1,1,1,1,1]            # weighting to apply to frames when generating heatmaps
HEAT_THRESHOLD = 20                             # min threshold for valid pixels

SHOW_INSERTS = True                             # whether to show the insert frames for info as to what's goin' on

class Frame():                                  # class object to hold video frame positive windows data
    def __init__(self):
        self.positive_windows = []

video_frames = []                               # will use this for a global store of previous frames

print('Starting video processing')
print()
svc, scaler = vd.load_classifier()
print('Classifier details:')
print(svc)
print(scaler)
print()

windows = []

def process_image(image_input):
    # takes an RGB image input, returns the final output with the lanes drawn

    # Advanced lane globals
    global n_count_since_update                     # filthy global variables, but allows use across frames
    global last_drawn_radius
    global last_drawn_offset
    global radius_rolling_count
    global offset_rolling_count
    global fFirstFrame
    global fGoodLastFrame
    global left_line
    global right_line

    # Vehicle finding globals
    global windows                              # filthy horrible globals
    global video_frames

    # Lane finding code...

    # Step 1: undistort the image, using the calibration data previous calculated for the camera
    image_undistort = AdvLanes.undistort_image(image_input, mtx, dist)

    # Step 2: get the binary thresholds
    binary_threshold = AdvLanes.preferred_threshold_image(image_undistort)

    # Step 3: change the perspective to overhead
    binary_overhead = AdvLanes.warp_original_to_overhead(binary_threshold)

    # Step 4 : find the lane lines
    if fFirstFrame or (fGoodLastFrame == False):
        # Always do the sliding-window approach on the first frame or after a failed frame
        left_fit, right_fit, leftx, lefty, rightx, righty, out_img = AdvLanes.find_lanes_sliding(binary_overhead)
    else:
        # Can try the search process, as we had a good fit on the last frame
        # Get the last fit parameters from the lists of line class objects
        last_left_fit = left_line[0].fit
        last_right_fit = right_line[0].fit
        left_fit, right_fit, leftx, lefty, rightx, righty = AdvLanes.find_lanes_search(binary_overhead, last_left_fit,
                                                                                       last_right_fit)
    # Now check if lines are parallel / if we have a good fit
    # Get the radii
    radius_avg, radius_l, radius_r = AdvLanes.get_curvature(leftx, lefty, rightx, righty)
    #print(radius_avg, radius_l, radius_r)
    if AdvLanes.check_similar_radii(radius_l, radius_r) or fFirstFrame:
        # radii are within the tolerance range of each other, so take this as reasonably parallel lines
        # just accept the first frame regardless, to get the process up and running...
        fGoodLastFrame = True
        # save this result
        new_left = Line()
        new_left.fit = left_fit
        new_left.points_x = leftx
        new_left.points_y = lefty
        left_line = [new_left] + left_line
        if len(left_line) > FRAMES_TO_KEEP:
            # pop the oldest, we only want to keep a small number of frames
            left_line.pop()
        new_right = Line()
        new_right.fit = right_fit
        new_right.points_x = rightx
        new_right.points_y = righty
        right_line = [new_right] + right_line
        if len(right_line) > FRAMES_TO_KEEP:
            right_line.pop()
        # can calculate the new radius and offset
        radius = radius_avg                             # use the average calculated initially
        offset = AdvLanes.get_offset(image_input.shape[1], left_fit, right_fit, leftx, lefty, rightx, righty)
    else:
        # radii are very different, so can't use this result - will just take a weighted rolling avergage instead
        fGoodLastFrame = False
        # get a smoothed values from the last 5 frames (or as many as we have so far)
        n_measurements_weight = 0
        fit_count = [0.0,0.0,0.0]
        idx = 0
        for idx in range(len(left_line)):
            fit_count = [fit_count[0] + (left_line[idx].fit[0] * SMOOTHING_WEIGHTS[idx]),
                         fit_count[1] + (left_line[idx].fit[1] * SMOOTHING_WEIGHTS[idx]),
                         fit_count[2] + (left_line[idx].fit[2] * SMOOTHING_WEIGHTS[idx])]
            n_measurements_weight += SMOOTHING_WEIGHTS[idx]
        left_fit = [fit_count[0]/n_measurements_weight, fit_count[1]/n_measurements_weight, fit_count[2]/n_measurements_weight]
        n_measurements_weight = 0
        fit_count = [0.0,0.0,0.0]
        idx = 0
        for idx in range(len(right_line)):
            fit_count = [fit_count[0] + (right_line[idx].fit[0] * SMOOTHING_WEIGHTS[idx]),
                         fit_count[1] + (right_line[idx].fit[1] * SMOOTHING_WEIGHTS[idx]),
                         fit_count[2] + (right_line[idx].fit[2] * SMOOTHING_WEIGHTS[idx])]
            n_measurements_weight += SMOOTHING_WEIGHTS[idx]
        right_fit = [fit_count[0]/n_measurements_weight, fit_count[1]/n_measurements_weight, fit_count[2]/n_measurements_weight]
        # leave the radius and offset the same as the last frame
        radius = last_drawn_radius
        offset = last_drawn_offset

    # Step 5 : draw the lane lines and polygon over a new image
    overhead_polygon = AdvLanes.plot_lanes_on_warped(binary_overhead, left_fit, right_fit)
    image_polygon = AdvLanes.warp_overhead_to_original(overhead_polygon)
    image_merged = AdvLanes.merge_over_camera_view(image_input, image_polygon)

    # # Step 6 : paste on the radius and curvature on the output image
    # # update the values every X frames, so it's read-able in the final video
    # if n_count_since_update == UPDATE_TEXT_INTERVAL:
    #     # update on this frame with then new value
    #     if not fFirstFrame:
    #         radius_rolling_count += radius
    #         radius_avg = radius_rolling_count / n_count_since_update
    #     else:
    #         radius_avg = radius
    #     if not fFirstFrame:
    #         offset_rolling_count += offset
    #         offset_avg = offset_rolling_count / n_count_since_update
    #     else:
    #         offset_avg = offset
    #     image_final = AdvLanes.add_text(image_merged, radius_avg, offset_avg)
    #     last_drawn_offset = offset_avg
    #     last_drawn_radius = radius_avg
    #     n_count_since_update = 0
    #     radius_rolling_count = 0
    #     offset_rolling_count = 0
    # else:
    #     # just use the last radius/offset measurement, so the values are readable in the video
    #     image_final = AdvLanes.add_text(image_merged, last_drawn_radius, last_drawn_offset)
    #     n_count_since_update += 1
    #     radius_rolling_count += radius
    #     offset_rolling_count += offset
    # fFirstFrame = False                                         # flip this flag after the first frame

    #final_image = image_merged

    # End of lane finding code - image_merged is the image with the lanes drawn

    # Vehicle finding code

    # Step 1: check the pixel value ranges - set to 0-1 if not already in that range
    img = np.copy(image_input)
    if np.max(img) > 1:
        img = img.astype(np.float32) / 255

    # quick test - simply run the basic classification on the frame

    # Step 2: create a list of sliding window boxes - various sizing - if we haven't already done so
    if len(windows) == 0:
        windows128 = vd.slide_window(img, xy_window=(128, 128), xy_overlap=(0.75, 0.75), y_start_stop=[350, 625])
        windows96 = vd.slide_window(img, xy_window=(96, 96), xy_overlap=(0.75, 0.75), y_start_stop=[350, 625])
        windows64 = vd.slide_window(img, xy_window=(64, 64), xy_overlap=(0.75, 0.75), y_start_stop=[350, 625])
        windows32 = vd.slide_window(img, xy_window=(32, 32), xy_overlap=(0.75, 0.75), y_start_stop=[350, 625])
        windows = windows128 + windows96 + windows64  # + windows32

    # Step 3: get the positive classification windows for this image
    car_windows = []
    for window in windows:
        # get the subset of the frame
        test_image = vd.get_window_image(img, window)
        # get the prediction
        prediction = vd.predict_window(test_image, svc, scaler, color_space=color_space_best,
                                       spatial_size=spatial_size_best,
                                       hist_bins=hist_bins_best, bins_range=(0, 1), orient=orient_best,
                                       pix_per_cell=pix_per_cell_best, cell_per_block=cell_per_block_best,
                                       hog_channel=hog_channel_best,
                                       spatial_feat=True, hist_feat=True, hog_feat=True)
        if prediction == 1.:
            # add the positive prediction to the list
            car_windows.append(window)

    new_frame = Frame()
    new_frame.positive_windows = car_windows
    video_frames = [new_frame] + video_frames
    if len(video_frames) > FRAMES_TO_KEEP:
        video_frames.pop()

    if SHOW_INSERTS:
        # create a view of the positive detections
        positives_img = vd.draw_boxes(image_input, car_windows)

    # Step 4: create a weighted heatmap from the most recent detections
    heatmap = np.zeros_like(img[:, :, 0])
    for i in range(0, len(video_frames)):
        # add the windows for this frame to the heatmap
        heatmap = vd.add_weighted_heat(heatmap,video_frames[i].positive_windows,FRAME_WEIGHTS[1])

    # Step 5: threshold the heatmap
    t_heatmap = vd.threshold_heatmap(heatmap, HEAT_THRESHOLD)

    # Step 6; get contiguous boxes
    cars_found = vd.get_label_boxes(t_heatmap)

    # Step 7: draw the boxes
    box_image = vd.draw_boxes(image_merged, cars_found)         # draw boxes on the merged image

    if SHOW_INSERTS:
        # convert heatmaps to colour, draw the inserts and label them - this is pretty slow...
        x_offset = 50
        y_offset = 50
        insert_width = 267
        insert_height = 150
        box_image = vd.overlay_main(box_image, positives_img, x_offset, y_offset, (insert_width,insert_height))
        box_image = vd.add_text(box_image, 'Detections', x_offset, 220)
        x_offset = x_offset + insert_width + 50
        c_heat = np.zeros_like(image_input) # heatmap.resize([heatmap.shape[0],heatmap.shape[1],3])
        for i in range(0, len(video_frames)):
            c_heat = vd.add_weighted_heat(c_heat, video_frames[i].positive_windows, FRAME_WEIGHTS[1])
        c_heat = cv2.applyColorMap(c_heat, cv2.COLORMAP_HOT)
        box_image = vd.overlay_main(box_image, c_heat ,x_offset, y_offset, (insert_width,insert_height))
        box_image = vd.add_text(box_image, 'Heatmap', x_offset, 220)
        t_heat = np.zeros_like(image_input)
        t_heat = vd.threshold_heatmap(c_heat, HEAT_THRESHOLD)
        t_heat = cv2.applyColorMap(t_heat, cv2.COLORMAP_HOT)
        x_offset = x_offset + insert_width + 50
        box_image = vd.overlay_main(box_image, t_heat, x_offset, y_offset, (insert_width,insert_height))
        box_image = vd.add_text(box_image, 'Thresholded heatmap', x_offset, 220)

    return box_image

# # process the test project video
# vid_output = 'test_video_output.mp4'
# clip1 = VideoFileClip("test_video.mp4", audio=False)         # get an error if don't expicitly turn off audio!
# vid_clip = clip1.fl_image(process_image)
# vid_clip.write_videofile(vid_output, audio=False)

# process the main project video
vid_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4", audio=False)         # get an error if don't expicitly turn off audio!
vid_clip = clip1.fl_image(process_image)
vid_clip.write_videofile(vid_output, audio=False)

# # process the challenge video
# n_count_since_update = UPDATE_TEXT_INTERVAL                     # init, so we start again
# last_drawn_radius = 0
# last_drawn_offset = 0
# radius_rolling_count = 0
# offset_rolling_count = 0
# fFirstFrame = True
# fGoodLastFrame = False
# left_line = []
# right_line = []
# vid_output = 'outputs/challenge_video_output.mp4'
# clip1 = VideoFileClip("challenge_video.mp4", audio=False)
# vid_clip = clip1.fl_image(process_image)
# vid_clip.write_videofile(vid_output, audio=False)
#
# # process the harder challenge video
# n_count_since_update = UPDATE_TEXT_INTERVAL                     # init, so we start again
# last_drawn_radius = 0
# last_drawn_offset = 0
# radius_rolling_count = 0
# offset_rolling_count = 0
# fFirstFrame = True
# fGoodLastFrame = False
# left_line = []
# right_line = []
# vid_output = 'outputs/harder_challenge_video_output.mp4'
# clip1 = VideoFileClip("harder_challenge_video.mp4", audio=False)
# vid_clip = clip1.fl_image(process_image)
# vid_clip.write_videofile(vid_output, audio=False)