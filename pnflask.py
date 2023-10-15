import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from flask import*
import os

app=Flask(__name__)
UPLOAD_FOLDER='static/data'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/')
@app.route("/bulk",methods=['GET','POST'])

def bulk():
    if request.method=="POST":
        file=request.files['data']
        na=file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        model = load_model('pneumonia_detection_model_final.h5')

        # Function to preprocess an image for prediction
        def load_and_preprocess_image(image_path, target_size):
            img = image.load_img(image_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array


        # Test image path
        test_image_path = r"static/data/"+na # Replace with the path of the test image

        # Preprocess the test image for prediction
        test_image = load_and_preprocess_image(test_image_path, target_size=(150, 150))  # Adjust the target size as needed
        # Make a prediction
        prediction = model.predict(test_image)

        # Interpret the prediction
        if prediction[0] > 0.5:
            os.remove(test_image_path)
            return render_template("index1.html",name="Pneumonia")
        else:
            os.remove(test_image_path)
            return render_template("index1.html",name="Normal")
        
    else:
        return render_template('index1.html',name="")
    
if __name__=='__main__':
    app.run(debug=True)
 
