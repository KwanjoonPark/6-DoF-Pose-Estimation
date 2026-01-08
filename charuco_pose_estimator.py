import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class CharucoPoseEstimator:
    def __init__(self):
        rospy.init_node('charuco_pose_estimator', anonymous=True)
        
        # 1. ChArUco Marker Infomation (User Settings)

        self.SQUARES_X = 4
        self.SQUARES_Y = 5
        self.SQUARE_LENGTH = 0.020  # meter
        self.MARKER_LENGTH = 0.015  # meter
        self.ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        
        # ChArUco Marker Generation
        self.board = aruco.CharucoBoard_create(
            self.SQUARES_X, self.SQUARES_Y, 
            self.SQUARE_LENGTH, self.MARKER_LENGTH, 
            self.ARUCO_DICT
        )

        # ROS Settings
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Basic topic of RealSense
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback)
        
        rospy.loginfo("Waiting for camera info...")

    def info_callback(self, msg):
        # Load RealSense camera parameter(Intrinsic) to ROS(one time only)
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera Info Received!")
            rospy.loginfo(f"Camera Matrix:\n{self.camera_matrix}")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return
        try:
            # ROS Image Message -> OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 1. Detect Marker
        corners, ids, rejected = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.parameters)

        # Debugging with Marker Image on Screen
        if ids is not None:
            aruco.drawDetectedMarkers(cv_image, corners, ids)

        if len(corners) > 0:
            # 2. 보간
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )

            if charuco_corners is not None and len(charuco_corners) > 0:
                # 3. Pose Estimation
                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board, 
                    self.camera_matrix, self.dist_coeffs, None, None
                )

                if valid:
                    # Draw Axis
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                    
                    # coordinates
                    pos_text = f"Pos(m): {np.round(tvec.T[0], 3)}"
                    cv2.putText(cv_image, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Convert coordinates to degree
                    x, y, z = tvec.flatten()
                    yaw = np.degrees(np.arctan2(x, z))
                    pitch = np.degrees(np.arctan2(y, z))
                    
                    angle_text = f"Angle(deg): Yaw={yaw:.2f}, Pitch={pitch:.2f}"
                    cv2.putText(cv_image, angle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # screen output
        cv2.imshow("ROS ChArUco Pose", cv_image)
        key = cv2.waitKey(1)
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = CharucoPoseEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()