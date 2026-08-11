import os
from flask import Flask, request, jsonify
import cv2
import ddddocr
import numpy as np

app = Flask(__name__)

# ddddocr ကို initialize လုပ်ခြင်း[cite: 1]
_ocr = ddddocr.DdddOcr(show_ad=False)

def ocr_processing(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, buffer = cv2.imencode('.png', thresh)
        result = _ocr.classification(buffer.tobytes())
        return result.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Python OCR Server is active and running!"}), 200

@app.route('/api/ocr/process', methods=['POST'])
def process_ocr():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    if not image_bytes:
        return jsonify({"success": False, "error": "Empty image file"}), 400
        
    extracted_text = ocr_processing(image_bytes)
    
    return jsonify({
        "success": True,
        "text": extracted_text
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
