# Image processing pipeline for vehicle detection video processing






def process_image(image_input):
    # takes an RGB image input, returns the final output with the lanes drawn


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
    box_image = vd.draw_boxes(image_input, cars_found)

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

# # process the test video
# vid_output = 'outputs/test_video_output.mp4'
# clip1 = VideoFileClip("test_video.mp4", audio=False)
# vid_clip = clip1.fl_image(process_image)
# vid_clip.write_videofile(vid_output, audio=False)

# process the project video
vid_output = 'outputs/project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4", audio=False)
vid_clip = clip1.fl_image(process_image)
vid_clip.write_videofile(vid_output, audio=False)