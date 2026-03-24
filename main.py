import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
from PIL import Image

app = FastAPI(title="KLE Tech Captcha Solver")

# Initialize PaddleOCR (PP-OCRv4 is the latest and best for this)
# use_angle_cls=True helps if the captcha text is slightly tilted
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def preprocess_captcha(image_bytes):
    # 1. Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Scale Up (4x) - Critical for 40px captchas
    # INTER_CUBIC is best for enlarging small text without pixelation
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # 4. Denoise & Binarize (Pure Black and White)
    # This removes the "noise" and "breaks" you were worried about
    _, binarized = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # PaddleOCR expects 3 channels (BGR)
    return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

@app.post("/solve-captcha")
async def solve_captcha(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_bytes = await file.read()
        
        # Preprocess to prevent "text breaking"
        processed_img = preprocess_captcha(image_bytes)
        
        # Perform OCR
        result = ocr.ocr(processed_img, cls=True)

        if not result or not result[0]:
            return {"captcha_text": "", "status": "no_text_found"}

        # Extract and clean the text
        # We take all detected blocks and join them, removing spaces
        raw_text = "".join([line[1][0] for line in result[0]])
        clean_text = raw_text.replace(" ", "").strip().upper()

        return {
            "captcha_text": clean_text,
            "confidence": result[0][0][1][1] if result[0] else 0.0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
