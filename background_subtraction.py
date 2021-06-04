import cv2 as cv
import numpy as np


'''
have the colour of the hand be different from the colour of the rest of the body, so that we can sample the
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

if __name__ == '__main__':
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
            
        fgMask = backSubtractor.apply(frame)

        # remove the background from the frame
        frame = cv.copyTo(src=frame, mask=fgMask)
        
        # search for the colour of the hand in this frame
        cv.imshow('Frame Masked', frame)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break