# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)


def draw_pose(image, landmarks):
    """
    TODO Task 1

    Code to this fucntion to draw circles on the landmarks and lines
    connecting the landmarks then return the image.

    Use the cv2.line and cv2.circle functions.

    landmarks is a collection of 33 dictionaries with the following keys
            x: float values in the interval of [0.0,1.0]
            y: float values in the interval of [0.0,1.0]
            z: float values in the interval of [0.0,1.0]
            visibility: float values in the interval of [0.0,1.0]

    References:
    https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    """

    # copy the image
    landmark_image = image.copy()

    # get the dimensions of the image
    height, width, _ = image.shape

    # Define landmark connections (pairs of indices to draw lines between)
    connections = [
        (12, 14), (14, 16), (16, 20), (16, 18), (20, 18), (16, 22), # Right arm
        (11, 13), (13, 15), (15, 19), (15, 17), (19, 17), (21, 15),  # Left arm
        (23, 24), (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),  # Right leg
        (23, 25), (25, 27), (27, 29), (29, 31), (31, 27), # Left leg
        (11, 12), (11, 23), (12, 24), (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (10, 9)  # Torso and head
    ]

    # Loop through landmarks and draw circles and lines
    for idx, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
  
        cv2.circle(landmark_image, (x, y), 5, (0, 255, 0), -1)

    # Draw lines between connected landmarks
    for connection in connections:
        start_landmark = landmarks.landmark[connection[0]]
        end_landmark = landmarks.landmark[connection[1]]
        
        # Convert landmark positions to pixel coordinates
        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))

        cv2.line(landmark_image, start_point, end_point, (255, 0, 0), 2)
   
    return landmark_image


def main():
    """
    TODO Task 2
            modify this fucntion to take a photo uses the pi camera instead
            of loading an image

    TODO Task 3
            modify this function further to loop and show a video
    """

    # Create a pose estimation model
    mp_pose = mp.solutions.pose

    # start detecting the poses
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        
        task2 = 0
        
        while not task2:
        
            # Capture the image from Pi camera
            image = pi_camera.capture_array()
            # Convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not
            # writeable to pass by reference.
            image.flags.writeable = False

            # get the landmarks
            results = pose.process(image)
            
            if results.pose_landmarks:
                image_with_landmarks = draw_pose(image, results.pose_landmarks)
            else:
                image_with_landmarks = image
            
            cv2.imshow("Pose Estimation", image_with_landmarks)
            

def test():
    # Create a pose estimation model
    mp_pose = mp.solutions.pose

    # start detecting the poses
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        # load test image
        image = cv2.imread("person.png")

        # To improve performance, optionally mark the image as not
        # writeable to pass by reference.
        image.flags.writeable = False

        # get the landmarks
        results = pose.process(image)

        if results.pose_landmarks != None:
            result_image = draw_pose(image, results.pose_landmarks)
            cv2.imwrite("output.png", result_image)
            print(results.pose_landmarks)
        else:
            print("No Pose Detected")


if __name__ == "__main__":
    main()
    print("done")
