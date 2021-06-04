import numpy as np
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while(True):
        # capture the frame
        ret, frame = cap.read()

        # resize and colour the frame
        frame = cv2.resize(frame, dsize=(300, 300), interpolation=cv2.INTER_NEAREST)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

