import numpy as np
import cv2
import os

RESIZE = 5
sample_area = ((0, 0), (0, 25), (25, 0), (25, 25))
sample_area_dimension = 25
sampled = False

def sample_skin(samples):
    '''
    Given a frame, returns the higher and lower threshold for finding the skin 
    colour of the user.
    '''
    high_threshold_offset = 10
    low_threshold_offset = 10

    sample_1_avg = samples[0].mean(axis=1).mean(axis=0)
    sample_2_avg = samples[1].mean(axis=1).mean(axis=0)
    sample_3_avg = samples[2].mean(axis=1).mean(axis=0)
    sample_4_avg = samples[3].mean(axis=1).mean(axis=0)

    sample_avg_left = np.mean(np.array([sample_1_avg, sample_3_avg]), axis=0)
    sample_avg_right = np.mean(np.array([sample_2_avg, sample_4_avg]), axis=0)

    '''
    This sampling uses the HSV format as to have more accurate results in changing lighting environments
    '''

    hLowThreshold = min(sample_avg_left[0], sample_avg_right[0]) - low_threshold_offset
    hHighThreshold = max(sample_avg_left[0], sample_avg_right[0]) + high_threshold_offset
    sLowThreshold = min(sample_avg_left[1], sample_avg_right[1]) - low_threshold_offset
    sHighThreshold = max(sample_avg_left[1], sample_avg_right[1]) + high_threshold_offset
    vLowThreshold = 0
    vHighThreshold = 255

    hsv_thresh = np.array([hLowThreshold, sLowThreshold, vLowThreshold, \
        hHighThreshold, sHighThreshold, vHighThreshold])
    return hsv_thresh



def backgroundRemover(current_frame, reference):
    '''
    Compares the current_frame to the reference, and if the pixels at the same position
    in the reference photo and current_frame are similar enough, then they are removed
    from the current_frame.
    '''
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    threshold = 30 # if the two pixels are different by a value of threshold, then remove
    for row in range(len(current_frame)):
        for column in range(len(current_frame[0])):
            reference_pixel = reference[row][column]
            current_frame_pixel = current_frame_gray[row][column] 
            # if the difference between the reference_pixel and the current_frame_pixel
            # is off by a value of threshold, remove it from the image
            if (reference_pixel - threshold <= current_frame_pixel) and \
                (reference_pixel + threshold >= current_frame_pixel):
                current_frame_gray[row][column] = 0
            else:
                current_frame_gray[row][column] = 255

    # apply the mask to the original
    return current_frame_gray


# Background subtraction methods
# cv2.BackgroundSubtractorMOG2

if __name__ == "__main__":
    if cv2.useOptimized is False:
        cv2.setUseOptimized(True)

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while(True):
        # capture the frame
        ret, frame = cap.read()
        frame_width = frame.shape[1] // RESIZE
        frame_height = int((frame.shape[0] / frame.shape[1]) * frame_width) 

        frame = cv2.resize(frame, dsize=(frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        foreground = backgroundRemover(frame, reference_frame)
        frame_hsv_bg_removed = cv2.copyTo(src=frame_hsv, mask=foreground)
        # draw our sampling area
        if sampled is False:
            for sample in sample_area:
                cv2.rectangle(frame, sample, (sample[0] + sample_area_dimension,\
                    sample[1] + sample_area_dimension), (255, 0, 255), 2)
        
        # if 's' is pressed, sample the colour of skin
        if (sampled is False) and (cv2.waitKey(1) & 0xFF == ord('s')):
            # sample the colour of the skin in the top left corner
            print("Sampling skin")
            samples = []
            # get all the sample squares
            for sample in sample_area:
                samples.append(frame_hsv[sample[0]:sample[0] + sample_area_dimension ,\
                    sample[1]:sample[1] + sample_area_dimension])
            # returns the average lowThreshold and highThreshold
            hsv_thresh = sample_skin(samples=samples)
            # binarize the image
            sampled = True

        # quit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('b'):
            reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (sampled):
            hsv_thresh_frame = cv2.inRange(frame_hsv_bg_removed, (hsv_thresh[0], hsv_thresh[1], hsv_thresh[2]),\
                 (hsv_thresh[3], hsv_thresh[4], hsv_thresh[5]))
            # open the image (erode then dilate) - removes noise
            ellipse_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            hsv_opening = cv2.morphologyEx(hsv_thresh_frame, cv2.MORPH_OPEN, ellipse_structure)
            # perform dilation to close any gaps
            hsv_dilated = cv2.dilate(hsv_opening, kernel=np.ones((3,3)), anchor=(-1, -1), iterations=3)
            cv2.imshow("hsv_frame", hsv_dilated)
        else:
            cv2.imshow('frame', frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()