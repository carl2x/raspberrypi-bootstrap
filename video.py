# Imports
import cv2
from picamera2 import Picamera2
import time


# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)


# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)


while True:
	# Get a image frame as a numpy array
	image = pi_camera.capture_array()


	# display the image
	cv2.imshow("Video", image)


	# This waits for 1 ms and if the 'q' key is pressed it breaks the loop	 
	if cv2.waitKey(1) == ord('q'):
		break
	
# Close all the windows
cv2.destroyAllWindows()


print('Done!')
