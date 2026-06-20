from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model("mobilenetv2_sawit.h5")

# Nama kelas
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

        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Tidak ada file yang dipilih"
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Silakan pilih gambar"
            )

        try:
            # Simpan file upload
            upload_folder = "static/uploads"

            filename = file.filename
            filepath = os.path.join(upload_folder, filename)

            file.save(filepath)

            # Baca gambar
            img = Image.open(filepath)
            img = img.convert("RGB")
            img = img.resize((224, 224))

            # Preprocessing
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_class = np.argmax(prediction)

            hasil = class_names[predicted_class]

            confidence = round(
                float(np.max(prediction)) * 100,
                2
            )

            image_path = filepath

        except Exception as e:
            return render_template(
                "index.html",
                error=str(e)
            )

    return render_template(
        "index.html",
        hasil=hasil,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
