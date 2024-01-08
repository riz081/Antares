from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
import uvicorn

app = FastAPI()

MODEL = tf.keras.models.load_model("./models/1")

CLASS_NAMES = ["Melon Sakit", "Melon Sehat"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/desease")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host = '192.168.1.6', port = 8000)