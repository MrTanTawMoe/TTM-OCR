import asyncio
import aiohttp
from aiohttp import web
import cv2
import ddddocr
import numpy as np
import os

# ddddocr ကို initialize လုပ်ခြင်း[cite: 1]
_ocr = ddddocr.DdddOcr(show_ad=False)

def _ocr_sync(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, buffer = cv2.imencode('.png', thresh)
    result = _ocr.classification(buffer.tobytes())
    return result

async def extract_text_from_image(image_bytes):
    return await asyncio.to_thread(_ocr_sync, image_bytes)

# ------------------- API HANDLER -------------------
async def handle_ocr_upload(request):
    try:
        reader = await request.multipart()
        image_bytes = None

        async for field in reader:
            if field.name == 'image':
                image_bytes = await field.read()
                break

        if not image_bytes:
            return web.json_response({"success": False, "error": "No image provided"}, status=400)

        # ပုံထဲက စာသားကို ddddocr ဖြင့် ဖတ်မည်[cite: 1]
        extracted_text = await extract_text_from_image(image_bytes)

        # APK ဆီသို့ ဖတ်လို့ရလာတဲ့ စာသား string ကို တိုက်ရိုက်ပြန်ပေးမည်
        return web.json_response({
            "success": True,
            "text": extracted_text
        })
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_root(request):
    return web.Response(text="OCR Image API Server is running!")

async def init_app():
    app = web.Application()
    app.router.add_get('/', handle_root)
    app.router.add_post('/api/ocr/extract', handle_ocr_upload)
    return app

if __name__ == '__main__':
    app = asyncio.run(init_app())
    port = int(os.environ.get('BOT_PORT', 8099))
    web.run_app(app, host='0.0.0.0', port=port)
