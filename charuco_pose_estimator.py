import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import csv
from datetime import datetime
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class CharucoPoseEstimator:
    def __init__(self):
        rospy.init_node('charuco_pose_estimator', anonymous=True)
        
        # 1. ChArUco Marker Infomation (User Settings)

        self.SQUARES_X = 5
        self.SQUARES_Y = 4
        self.SQUARE_LENGTH = 0.02  # 2.0 cm -> 0.02 m (OpenCV uses meters internally)
        self.MARKER_LENGTH = 0.015  # 1.5 cm -> 0.015 m
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

        # Storage for current frame data
        self.current_image = None
        self.current_pose_data = None
        self.save_counter = 0

        # Create directories for saving
        self.images_dir = "images"
        self.csv_path = "self_pose.csv"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            rospy.loginfo(f"Created directory: {self.images_dir}")

        # Initialize CSV file with header if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Image_File', 'Distance(cm)', 'Pitch(deg)', 'Yaw(deg)', 'Roll(deg)', 'X(cm)', 'Y(cm)', 'Z(cm)'])
            rospy.loginfo(f"Created CSV file: {self.csv_path}")

        # Basic topic of RealSense
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback)

        rospy.loginfo("Waiting for camera info...")
        rospy.loginfo("Press 's' to save current image and pose data")

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

        # Store original image for potential saving
        self.current_image = cv_image.copy()

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
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

                    # coordinates (tvec is in meters, convert to cm)
                    pos_cm = tvec.T[0] * 100  # m -> cm
                    pos_text = f"Pos(cm): X={pos_cm[0]:.2f}, Y={pos_cm[1]:.2f}, Z={pos_cm[2]:.2f}"
                    cv2.putText(cv_image, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Euclidean distance from camera
                    distance_cm = np.linalg.norm(tvec) * 100  # m -> cm
                    dist_text = f"Distance(cm): {distance_cm:.2f}"
                    cv2.putText(cv_image, dist_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Convert rotation vector to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)

                    # Calculate Euler angles using scipy convention (intrinsic XYZ)
                    # This method is more reliable for OpenCV coordinate system

                    # Extract angles from rotation matrix
                    # Using ZYX Euler angles (more common in robotics)
                    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

                    singular = sy < 1e-6

                    if not singular:
                        # Non-singular case
                        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))   # RX corresponds to Pitch
                        ry = np.degrees(np.arctan2(-R[2, 0], sy))      # RY corresponds to Yaw
                        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))    # RZ corresponds to Roll
                    else:
                        # Gimbal lock case
                        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                        ry = np.degrees(np.arctan2(-R[2, 0], sy))
                        rz = 0

                    # Normalize RX (Pitch) to be 0 when board faces camera
                    # RX is around ±180 when facing camera, so we normalize it
                    # Forward tilt → negative, Backward tilt → positive
                    if rx > 90:
                        pitch = rx - 180  # Convert 180 to 0, 90 to -90
                    elif rx < -90:
                        pitch = rx + 180  # Convert -180 to 0, -90 to 90
                    else:
                        pitch = rx

                    yaw = ry
                    roll = rz

                    angle_text = f"Angle(deg): Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}"
                    cv2.putText(cv_image, angle_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Store current pose data for potential saving
                    self.current_pose_data = {
                        'distance': distance_cm,
                        'pitch': pitch,
                        'yaw': yaw,
                        'roll': roll,
                        'x': pos_cm[0],
                        'y': pos_cm[1],
                        'z': pos_cm[2]
                    }

        # screen output
        cv2.imshow("ROS ChArUco Pose", cv_image)
        key = cv2.waitKey(1)

        # Handle 's' key press to save data
        if key == ord('s'):
            self.save_current_data()

    def save_current_data(self):
        """Save current image and pose data when 's' key is pressed"""
        if self.current_image is None:
            rospy.logwarn("No image available to save")
            return

        if self.current_pose_data is None:
            rospy.logwarn("No pose data available to save. Make sure marker is detected.")
            return

        # Generate filename with timestamp and counter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_counter += 1
        image_filename = f"image_{timestamp}_{self.save_counter:03d}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)

        # Save original image
        cv2.imwrite(image_path, self.current_image)
        rospy.loginfo(f"✅ Saved image: {image_path}")

        # Append pose data to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                image_filename,
                f"{self.current_pose_data['distance']:.2f}",
                f"{self.current_pose_data['pitch']:.2f}",
                f"{self.current_pose_data['yaw']:.2f}",
                f"{self.current_pose_data['roll']:.2f}",
                f"{self.current_pose_data['x']:.2f}",
                f"{self.current_pose_data['y']:.2f}",
                f"{self.current_pose_data['z']:.2f}"
            ])
        rospy.loginfo(f"✅ Saved pose data to: {self.csv_path}")
        rospy.loginfo(f"   Distance: {self.current_pose_data['distance']:.2f}cm, Pitch: {self.current_pose_data['pitch']:.1f}°, Yaw: {self.current_pose_data['yaw']:.1f}°, Roll: {self.current_pose_data['roll']:.1f}°")

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