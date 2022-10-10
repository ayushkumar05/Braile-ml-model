from flask import Flask,render_template,request,jsonify
import os
from keras.models import load_model
import numpy as np
import cv2
from keras_preprocessing import image


app = Flask(__name__)
model = load_model('Flask/model1.h5')
UPLOAD_FOLDER = 'Flask/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def ProcessInput(image):
    image = cv2.resize(image, (28,28))
    image = image.reshape(-1,28,28,3)
    pred = model.predict(image)
    prediction = np.argmax(pred)
    return prediction

@app.route("/",methods = ["GET","POST"])
def lcd():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        img = cv2.imread(path)
        #img = image.load_img(path, target_size=(224, 224))
        return render_template('tab.html',DATA = ProcessInput(img))


    return render_template('tab.html')

if __name__ == '__main__':
    app.run(debug=True)