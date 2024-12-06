from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

try:
    model = YOLO("yolov8n.pt")  
    print("YOLO model initialized successfully.")
except Exception as e:
    print(f"Error initializing YOLO model: {e}")
    model = None  


def detect_object(image_path, category):
    category_map = {
        'car': 2,
        'truck': 7,
        'bicycle': 1,
        'motorcycle': 3
    }

    if category not in category_map:
        raise ValueError(f"Unsupported category: {category}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load the image from the path: {image_path}")

    results = model(image, classes=[category_map[category]], conf=0.5, verbose=False)

    for r in results:
        if r.boxes: 
            for box in r.boxes:
                cls = int(box.cls[0])  # Class index
                if cls == category_map[category]:
                    return True

    return False


@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        category = request.form.get('category', 'truck')
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        result = detect_object(file_path, category)
        os.remove(file_path)
        return jsonify({"result": result}), 200

    except Exception as e:
        app.logger.error(f"Error during detection: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if model is None:
        print("Exiting application due to YOLO initialization failure.")
    else:
        os.makedirs("uploads", exist_ok=True)
        app.run(debug=True, port=4040)
