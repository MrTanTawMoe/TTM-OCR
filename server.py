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
        # ddddocr က raw bytes ကို တိုက်ရိုက်ဖတ်တာ ပိုကောင်းပါတယ်
        # OpenCV နဲ့ thresholding လုပ်တာကို ဖြုတ်လိုက်ပါမယ်
        result = _ocr.classification(image_bytes)
        return result.upper().strip() # စာလုံးကြီးနဲ့ပဲ ပြန်ပေးမယ်
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
