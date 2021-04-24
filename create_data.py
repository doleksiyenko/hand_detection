import numpy as np
import cv2
import os

RESIZE = 5

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while(True):
        # capture the frame
        ret, frame = cap.read()
        frame_width = frame.shape[1] // RESIZE
        frame_height = int((frame.shape[0] / frame.shape[1]) * frame_width) 

        frame = cv2.resize(frame, dsize=(frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (3,3), 0)

        ret, thresh = cv2.threshold(frame_gray, 128, 255, cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # hull = cv2.convexHull(contours)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        # cv2.drawContours(frame, hull, -1, (0, ,0), 3)
        cv2.imshow('frame', frame)


        # quit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()