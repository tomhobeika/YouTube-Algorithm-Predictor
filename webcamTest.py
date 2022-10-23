import cv2 as cv
from tensorflow import keras
import numpy as np

# Load pretrained model
model = keras.models.load_model('model.h5')

# Get image size from the input
img_size = model.input_shape[0][1:3]

cap = cv.VideoCapture(0)
while True:
	ret, img = cap.read()

	# Resize image to model input size
	thumb = cv.resize(img, (img_size[1], img_size[0]))
	# Convert from BGR to RGB and normalize 0-1
	thumb = cv.cvtColor(thumb, cv.COLOR_BGR2RGB) / 255
	
	# Predict views
	thumb = np.array([thumb])
	subs = np.array([0.1])
	pred = np.argmax(model.predict([thumb, subs], verbose=0), axis=1)[0]

	cv.putText(img=img, text=f"{10 ** pred} views", fontFace=cv.FONT_HERSHEY_SIMPLEX, org=(16, 32), fontScale=1, color=(255, 255, 255), thickness=2)

	cv.imshow('Webcam AI Demo', img)
	if cv.waitKey(1) == ord('q'):
		break
cap.release()
cv.destroyAllWindows()