from flask import *
import os
from werkzeug.utils import secure_filename
import Prediction_of_images


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        final_result = Prediction_of_images.prediction_food_calories(file_path)
        result = final_result.splitlines()
        os.remove(file_path)
        return result


if __name__ == '__main__':
    app.run()
