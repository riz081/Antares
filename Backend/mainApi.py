from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pandas as pd
import requests
from io import BytesIO

app = FastAPI()

# Load the trained SVM model
tanah_dataset = pd.read_csv('dummyData.csv')
X = tanah_dataset.drop(columns='Output', axis=1)
Y = tanah_dataset['Output']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

classifier = svm.SVC(kernel='linear')
classifier.fit(X, Y)

MODEL = tf.keras.models.load_model("./models/1")

CLASS_NAMES = ["Melon Sakit", "Melon Sehat"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def fetch_antares_data(api_url, headers):
    try:
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch data. Status code: {response.status_code}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def extract_con_values(data):
    try:
        # Sort the data based on the timestamp in descending order
        sorted_data = sorted(data.get('m2m:list', []), key=lambda x: x.get('m2m:cin', {}).get('ct', ''), reverse=True)
        
        # Extract "con" values from the latest entry
        con_values = [sorted_data[0].get('m2m:cin', {}).get('con', 'N/A')]
        con_values_int = [int(value.strip("'")) for value in con_values]
        return con_values_int
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting con values: {str(e)}")

@app.post('/classification')
async def predict():
    try:
        # Fetch the newest "con" value from the Antares APIs
        api_urls = [
            'https://platform.antares.id:8443/~/antares-cse/antares-id/RooftopITTS2/Nitrogen2?fu=1&drt=2&ty=4',
            'https://platform.antares.id:8443/~/antares-cse/antares-id/RooftopITTS2/pH2?fu=1&drt=2&ty=4',
            'https://platform.antares.id:8443/~/antares-cse/antares-id/RooftopITTS2/Potassium2?fu=1&drt=2&ty=4',
            'https://platform.antares.id:8443/~/antares-cse/antares-id/RooftopITTS2/Phosporus2?fu=1&drt=2&ty=4',
            'https://platform.antares.id:8443/~/antares-cse/antares-id/RooftopITTS2/Moisture2?fu=1&drt=2&ty=4',
            'https://platform.antares.id:8443/~/antares-cse/antares-id/RooftopITTS2/Temperature2?fu=1&drt=2&ty=4',
            'https://platform.antares.id:8443/~/antares-cse/antares-id/RooftopITTS2/Conductivity2?fu=1&drt=2&ty=4'
        ]
        
        headers = {'X-M2M-Origin': 'dfadb386eb62b10a:99882941cb61d872', 'X-M2M-Key': 'your_password'}
        
        con_values = []
        
        for api_url in api_urls:
            antares_data = fetch_antares_data(api_url, headers)

            if antares_data:
                # Extract newest "con" value
                newest_con_value = extract_con_values(antares_data)[0]
                con_values.append(newest_con_value)
            else:
                raise HTTPException(status_code=500, detail='Failed to fetch Antares data')

        # Perform prediction with the newest "con" values
        std_data = scaler.transform([con_values])
        prediction = classifier.predict(std_data)
        
        result = 'Buruk' if prediction[0] == 0 else ('Kurang Subur' if prediction[0] == 1 else ('Subur' if prediction[0] == 2 else 'Sangat Subur'))
        
        return {'forecasting': result, 'data_con': con_values}

    except HTTPException as e:
        raise e

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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host = '192.168.1.7', port = 8000)
