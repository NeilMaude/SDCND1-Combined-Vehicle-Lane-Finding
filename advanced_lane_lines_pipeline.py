# Run the image processing pipeline on video

import advanced_lane_lines as AdvLanes                          # main project code

# Video processing tools
import imageio.plugins
imageio.plugins.ffmpeg.download()                               # get the MPEG codec if not already present
from moviepy.editor import VideoFileClip

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

def process_image(image_input):
    # takes an RGB image input, returns the final output with the lanes drawn

    global n_count_since_update                     # filthy global variables, but allows use across frames
    global last_drawn_radius
    global last_drawn_offset
    global radius_rolling_count
    global offset_rolling_count
    global fFirstFrame
    global fGoodLastFrame
    global left_line
    global right_line

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

    final_image = image_merged
    return final_image

# process the test project video
vid_output = 'test_video_output.mp4'
clip1 = VideoFileClip("test_video.mp4", audio=False)         # get an error if don't expicitly turn off audio!
vid_clip = clip1.fl_image(process_image)
vid_clip.write_videofile(vid_output, audio=False)

# # process the main project video
# vid_output = 'outputs/project_video_output.mp4'
# clip1 = VideoFileClip("project_video.mp4", audio=False)         # get an error if don't expicitly turn off audio!
# vid_clip = clip1.fl_image(process_image)
# vid_clip.write_videofile(vid_output, audio=False)

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