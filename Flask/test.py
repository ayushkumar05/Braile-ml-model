import numpy as np
from keras.models import load_model
import cv2

model = load_model('./model1.h5')
image = cv2.imread("./abc.jpg")
image = cv2.resize(image, (28,28))
image = image.reshape(-1,28,28,3)

pred = model.predict(image)

prediction = np.argmax(pred)

print(prediction)