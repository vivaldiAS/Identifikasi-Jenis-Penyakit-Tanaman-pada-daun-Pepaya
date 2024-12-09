from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

app = Flask(__name__)

# Direktori konfigurasi
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/pepaya_disease_model.keras'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder upload tersedia
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)
CLASS_NAMES = ['Antraknosa', 'Blackspot', 'Keriting-Daun']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Simpan file upload
        file = request.files['file']
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Prediksi gambar
            img = load_img(filepath, target_size=(224, 224))  # Sesuaikan target_size
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)

            # Debugging output
            print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

            # Return the results and display modal
            return render_template('result.html', 
                                   filename=file.filename, 
                                   predicted_class=predicted_class, 
                                   confidence=confidence)

    return render_template('index.html', show_modal=False)

# Route untuk menampilkan gambar
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
