import cv2, sys


class Camera():

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        sys.exit("Cannot open camera") if not self.cap.isOpened() else print("Here We Go")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            sys.exit("Can't receive frame (stream end?). Exiting ...")

        return frame

    def release_cam(self):
        self.cap.release()