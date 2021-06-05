import cv2 as cv
import numpy as np
import os


def sample_skin(samples):
    '''
    Given a frame, returns the higher and lower threshold for finding the skin 
    colour of the user.
    '''
    high_threshold_offset = 20
    low_threshold_offset = 20

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

    hsv_thresh = np.array([(hLowThreshold, sLowThreshold, vLowThreshold), \
        (hHighThreshold, sHighThreshold, vHighThreshold)])
    return hsv_thresh


def clean_image(image):
    '''
    Cleans up a grayscale image by applying opening then closing.
    '''
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    # opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    closing_kernel = np.ones((6, 6), np.uint8)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    # blurred = cv.blur(closing, (3,3))
    return closing

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open camera.')
    exit(0)

backSubtractor = cv.createBackgroundSubtractorKNN()
faceDetection = cv.CascadeClassifier()
sampled = False
sample_area_dimension = 25
sampling_area = ((0, 0), (0, sample_area_dimension), (sample_area_dimension, 0), (sample_area_dimension, sample_area_dimension))

'''
Have the colour of the hand be different from the colour of the rest of the body, so that we can sample the
colour of the hand, and after applying the background subtractor, we search for the colour of the hand in this
image.

after isolating the hand, we can find the convex hull / rectangle around this hand, and then crop this image, and sample it
'''

if __name__ == '__main__':
    if not os.path.isdir('dataset'):
        os.mkdir('dataset')
        print("Created /dataset")
    
    os.chdir('dataset')
    print("Navigating to /dataset")

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame = cv.GaussianBlur(frame, (5, 5), 1)
        cv.imshow('Gaussian Blur', frame)
        frame_hsv_bgrd = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  
        fgMask = backSubtractor.apply(frame)

        # remove the background from the frame
        frame = cv.copyTo(src=frame, mask=fgMask)
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # press 'q' to quit the program
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # press 'c' to capture and save a sample image
        if (sampled is True) and (cv.waitKey(1) & 0xFF == ord('c')):
            cv.imwrite(os.getcwd() + '/test.png', frame_hsv_threshold)
            print(os.getcwd())

        # press 's' to sample the colour of the skin within the bounded area
        if (sampled is False) and (cv.waitKey(1) & 0xFF == ord('s')):
            print("Sampling colour of glove.")
            samples = []
            for sample in sampling_area:
                samples.append(frame_hsv[sample[0]:sample[0] + sample_area_dimension ,\
                    sample[1]:sample[1] + sample_area_dimension])
            # returns the average lowThreshold and highThreshold
            hsv_thresh = sample_skin(samples=samples)
            sampled = True
        
        if sampled is False:
            for sample in sampling_area:
                cv.rectangle(frame, sample, (sample[0] + sample_area_dimension,\
                    sample[1] + sample_area_dimension), (255, 0, 0), thickness=2)
                cv.imshow('Frame Masked', frame)

        if sampled is True:
            frame_hsv_threshold = cv.inRange(frame_hsv, hsv_thresh[0], hsv_thresh[1])

            cleaned_frame = clean_image(frame_hsv_threshold)

            contours, hierarchy = cv.findContours(cleaned_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # find the largest contour by area
            max_contour = max(contours, key=cv.contourArea) 
            x, y, w, h = cv.boundingRect(max_contour) # find the bounding box of the max contour
            cv.rectangle(cleaned_frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2) # draw a rectangle on the frame

            # create a frame that is the cropped version of frame, only inside the bounding box
            bounding_box_frame = frame_hsv_threshold[y: y + h, x: x + w]

            cv.imshow('Cleaned image', cleaned_frame)
            # cv.imshow('Frame Masked', frame_hsv_threshold)
            # cv.imshow('Bounding Box Frame', bounding_box_frame)

    capture.release()
    cv.destroyAllWindows()