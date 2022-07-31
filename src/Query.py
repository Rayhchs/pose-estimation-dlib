import cv2, os
import dlib
from scipy.spatial.transform import Rotation as R
import numpy as np


class Query():

    def __init__(self, frame):
        self.frame = frame
        self.size = self.frame.shape

    def open(self):
        cv2.putText(self.frame, '----- Press "s" to start -----', (int(self.size[1]/2-200), self.size[0]-25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.frame, '----- Press "q" to quit  -----', (int(self.size[1]/2-200), self.size[0]-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return self.frame

    def pose_estimation(self, detector, predictor, count=0, if_record=False):

        origin_frame = self.frame.copy()
        cv2.putText(self.frame, '----- Press "v" to record -----', (int(self.size[1]/2-200), self.size[0]-25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.frame, '----- Press "q" to quit  -----', (int(self.size[1]/2-200), self.size[0]-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Camera internals
        size = self.frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )

        # Find 68 points from dlib
        ps = []
        rects = detector(self.frame, 0)
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(self.frame,rects[i]).parts()])
            for idx, point in enumerate(landmarks):

                # 68 points
                pos = (point[0, 0], point[0, 1])
                ps.append(pos)

        # Capture one face
        if len(ps) == 68: 

            # Find 2D face points
            image_points = np.array([
                                    ps[30],     # Nose tip
                                    ps[8],      # Chin
                                    ps[36],     # Left eye left corner
                                    ps[45],     # Right eye right corner
                                    ps[48],    # Left mouth corner
                                    ps[54]     # Right mouth corner
                                ], dtype="double")

            # 3D model points.
            model_points = np.array([(0.0, 0.0, 0.0),
                                    (0.0, -300.0, -65.0),
                                    (-150.0, 170.0, -135.0),
                                    (150.0, 170.0, -135.0),
                                    (-150.0, -150.0, -125.0),
                                    (150.0, -150.0, -125.0)])

            for p in image_points:
                cv2.circle(self.frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

            # Calculate rotation vector to euler angle
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)#, flags=cv2.CV_ITERATIVE)
            rv = np.array((rotation_vector[0][0], rotation_vector[1][0], rotation_vector[2][0]))
            r = R.from_rotvec(rv)
            angle = r.as_euler('zyx', degrees=True) # calculate euler angle

            # Print text to image
            cv2.putText(self.frame, 'x: '+ str(round(angle[0],2)), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.frame, 'y: '+ str(round(angle[1],2)), (0, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.frame, 'z: '+ str(round(angle[2],2)), (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Exclude multi-face
        elif len(ps) > 69:
            cv2.putText(self.frame, 'Warning! There are too many people', (int(size[1]/2-200), int(size[0]/2)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Record image or not
        if if_record:

            cv2.putText(self.frame, '-----      Recording     -----', (int(self.size[1]/2-200), self.size[0]-80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(self.frame, '----- Press "p" to pause -----', (int(self.size[1]/2-200), self.size[0]-60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

            if len(ps) == 68:
                for p in image_points:
                    cv2.circle(origin_frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

                # Print text to image
                cv2.putText(origin_frame, 'x: '+ str(round(angle[0],2)), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(origin_frame, 'y: '+ str(round(angle[1],2)), (0, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(origin_frame, 'z: '+ str(round(angle[2],2)), (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            self.record(count, origin_frame)

        return self.frame


    def record(self, count, frame):
        os.mkdir('../demo') if not os.path.exists('../demo') else None
        k = '%03d' % count
        cv2.imwrite(f'../demo/{k}.jpg', frame)