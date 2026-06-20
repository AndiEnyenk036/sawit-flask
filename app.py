from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Inisialisasi Flask
app = Flask(__name__)

# Load model
model = load_model("mobilenetv2_sawit.h5")

# Nama kelas sesuai urutan di Colab
class_names = [
    "Fusarium",
    "Magnesium",
    "Manganese",
    "Potassium",
    "Rachis",
    "Sehat",
    "bercak"
]

@app.route("/", methods=["GET", "POST"])
def home():

    hasil = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":

        # Cek file
        if "image" not in request.files:
            error = "Tidak ada file yang dipilih"
            return render_template(
                "index.html",
                error=error
            )

        file = request.files["image"]

        if file.filename == "":
            error = "Silakan pilih gambar"
            return render_template(
                "index.html",
                error=error
            )

        # Folder upload
        upload_folder = os.path.join("static", "uploads")

if not os.path.isdir(upload_folder):
    os.makedirs(upload_folder)

        filepath = os.path.join(
            upload_folder,
            file.filename
        )

        file.save(filepath)

        try:

            # Baca gambar
            img = Image.open(filepath)
            img = img.convert("RGB")
            img = img.resize((224, 224))

            # Ubah ke numpy
            img_array = np.array(img)

            # Tambah batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Prediksi
            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_class = np.argmax(
                prediction
            )

            hasil = class_names[
                predicted_class
            ]

            confidence = round(
                float(np.max(prediction)) * 100,
                2
            )

            image_path = filepath

        except Exception as e:

            error = f"Error: {str(e)}"

            return render_template(
                "index.html",
                error=error
            )

    return render_template(
        "index.html",
        hasil=hasil,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
