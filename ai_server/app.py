from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load the YOLOv8 model
MODEL_PATH = os.getenv('MODEL_PATH', 'yolov8n.pt')
model = YOLO(MODEL_PATH)

# Waste category mapping
CATEGORY_MAP = {
    'plastic': 'recyclable',
    'bottle': 'recyclable',
    'can': 'recyclable',
    'paper': 'recyclable',
    'cardboard': 'recyclable',
    'glass': 'recyclable',
    'metal': 'recyclable',
    'food': 'compostable',
    'organic': 'compostable',
    'fruit': 'compostable',
    'vegetable': 'compostable',
    'battery': 'hazardous',
    'electronic': 'hazardous',
    'chemical': 'hazardous',
}

def map_category(label):
    label_lower = label.lower()
    for key, category in CATEGORY_MAP.items():
        if key in label_lower:
            return category
    return 'landfill'

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': MODEL_PATH})
@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference
        results = model(image)
        
        if not results or len(results) == 0:
            return jsonify({'error': 'No objects detected'}), 400
        
        # Get the top detection
        result = results[0]
        if len(result.boxes) == 0:
            return jsonify({'error': 'No objects detected'}), 400
        
        # Get the highest confidence detection
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        
        top_idx = confidences.argmax()
        top_confidence = float(confidences[top_idx])
        top_class = int(classes[top_idx])
        
        # Get class name
        class_name = result.names[top_class]
        category = map_category(class_name)
        
        return jsonify({
            'name': class_name,
            'category': category,
            'confidence': top_confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)