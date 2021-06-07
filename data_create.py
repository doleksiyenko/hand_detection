import sys
import cv2 as cv
import numpy as np
from skimage.transform import rescale
import os

import tensorflow as tf


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
    Cleans up a grayscale image by applying closing.
    '''
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    return closing


def resize_image(image):
    '''
    Resize image to 144p: (256, 144)
    '''
    dimension = image.shape
    
    height_diff = 144 - (dimension[0] % 144) # how far is the height from the next multiple of 144
    width_diff = 256 - (dimension[1] % 256) # how far is the width from the next multiple of 256

    # change the height and width so that they are multiples of 256 and 144 respectively
    target_width = dimension[1] + width_diff
    target_height = dimension[0] + height_diff

    # scale the image down to get size of (256, 144)
    scale_factor_w = target_width / 256
    scale_factor_h = target_height / 144

    scale_factor = scale_factor_w

    if (scale_factor_w != scale_factor_h):
        max_scale_factor = max(scale_factor_w, scale_factor_h)
        min_scale_factor = min(scale_factor_w, scale_factor_h)

        if min_scale_factor == scale_factor_w:
            target_width *= (max_scale_factor / min_scale_factor)
        else:
            target_height *= (max_scale_factor / min_scale_factor)

        scale_factor = max_scale_factor
    
    new_frame = np.zeros((int(target_height), int(target_width)))
    new_frame[:dimension[0], :dimension[1]] = image   
    resized_image = rescale(new_frame, (1 / scale_factor), anti_aliasing=False)

    return resized_image


def save_image(image, name):
    '''
    Save the current bounding box frame <image> to the directory /dataset
    '''
    # transform the image into 16:9 aspect ratio
    resized_image = resize_image(image) 
    # save the image into the current directory
    cv.imwrite(os.getcwd() + f'/{name}.png', resized_image)
    

def predict_gesture():
    '''
    Return the predicted gesture given by the trained TF model.
    '''
    pass


'''
Have the colour of the hand be different from the colour of the rest of the body, so that we can sample the
colour of the hand, and after applying the background subtractor, we search for the colour of the hand in this
image.

after isolating the hand, we can find the convex hull / rectangle around this hand, and then crop this image, and sample it
'''

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open camera.')
    exit(0)

backSubtractor = cv.createBackgroundSubtractorKNN()
faceDetection = cv.CascadeClassifier()
sampled = False
sample_area_dimension = 25
sampling_area = ((0, 0), (0, sample_area_dimension), (sample_area_dimension, 0), (sample_area_dimension, sample_area_dimension))

if __name__ == '__main__':
    # provide the name of which images the program is creating as an argument
    if (len(sys.argv) != 2):
        print("Pass in the name of gesture which program will capture.")
        exit(1)

    file_name = sys.argv[1]
    print(f"Will save files as {file_name}.")

    if not os.path.isdir('dataset'):
        os.mkdir('dataset')
        print('Created /dataset')
    
    os.chdir('dataset')
    print('Navigating to /dataset')

    # load in the model
    model = tf.keras.models.load_model('2-gesture-CNN.model')

    if not os.path.isdir(f'{file_name}'):
        os.mkdir(f'{file_name}')
        print(f'Created /{file_name}')
    
    os.chdir(f'{file_name}')
    print(f'Navigating to {file_name}')

    image_number = 0 # for the file name being saved
    captured_iteration = 0 # how many iterations of while loop have been completed (for capturing multiple images)
    capturing = False

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame = cv.GaussianBlur(frame, (5, 5), 1)
        frame_hsv_bgrd = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  
        fgMask = backSubtractor.apply(frame)

        # remove the background from the frame
        frame = cv.copyTo(src=frame, mask=fgMask)
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # press 'q' to quit the program
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # press 'c' to capture a single image
        if (sampled is True) and (cv.waitKey(1) & 0xFF == ord('c')):
            save_image(bounding_box_frame, f'{file_name}_{image_number}')
            image_number += 1
            print("Saved into " + os.getcwd())
        
        # press 'n' to capture 1000 images
        if (sampled is True) and (cv.waitKey(1) & 0xFF == ord('n')):
            capturing = True

           
        if captured_iteration == 1000:
            capturing = False
            captured_iteration = 0

        if capturing is True:
            save_image(bounding_box_frame, f'{file_name}_{image_number}')
            image_number += 1
            captured_iteration += 1
            print(f'Capturing {captured_iteration}/1000', end='\r')
        
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

            
            bounding_box_frame = cleaned_frame[y: y + h, x: x + w]

            cv.imshow('Cleaned image', cleaned_frame)
            cv.imshow('Bounding Box Frame', bounding_box_frame)

            resized_bounding_frame = resize_image(bounding_box_frame)
            resized_bounding_frame = resized_bounding_frame.reshape(-1, 144, 256, 1)

            gesture_prediction = model.predict([resized_bounding_frame])
            score = tf.nn.softmax(gesture_prediction[0])

            if score[0] > 0.7:
                print("palm")
            if score[1] > 0.7:
                print('thumb')

            # print(score)

    capture.release()
    cv.destroyAllWindows()