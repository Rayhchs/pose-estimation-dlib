from Camera import Camera
from Query import Query
import cv2, dlib, sys


def main():

    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
    except:
        sys.exit("You should download 'shape_predictor_68_face_landmarks.dat' first")

    camera = Camera()
    state = 0
    count = 0
    while True:
        frame = camera.get_frame()
        user_command = Query(frame)

        # Confirm key input
        keycode = cv2.waitKey(1)
        if keycode == ord('q'):
            break
        if keycode == ord('s') and state == 0:
            state = 1
        if keycode == ord('v') and state == 1:
            state = 2
        if keycode == ord('p') and state == 2:
            state = 1

        # Interaction
        if state == 0:
            frame = user_command.open()
        elif state == 1:
            frame = user_command.pose_estimation(detector, predictor)
        elif state == 2:
            frame = user_command.pose_estimation(detector, predictor, count, if_record=True)
            count += 1

        cv2.imshow("Pose Estimation", frame)

    camera.release_cam()


if __name__ == '__main__':
    main()