from flask import Flask, jsonify, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pandas as pd
import requests

app = Flask(__name__)

# Load the trained SVM model
tanah_dataset = pd.read_csv('dataset2.csv')
X = tanah_dataset.drop(columns='Output', axis=1)
Y = tanah_dataset['Output']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

classifier = svm.SVC(kernel='linear')
classifier.fit(X, Y)

def fetch_antares_data(api_url, headers):
    try:
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def extract_con_values(data):
    try:
        # Sort the data based on the timestamp in descending order
        sorted_data = sorted(data.get('m2m:list', []), key=lambda x: x.get('m2m:cin', {}).get('ct', ''), reverse=True)
        
        # Extract "con" values from the latest entry
        con_values = [sorted_data[0].get('m2m:cin', {}).get('con', 'N/A')]
        con_values_int = [int(value.strip("'")) for value in con_values]
        return con_values_int
    except Exception as e:
        print(f"Error extracting con values: {str(e)}")
        return []

@app.route('/classification', methods=['POST'])
def predict():
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
                return jsonify({'error': 'Failed to fetch Antares data'})

        # Perform prediction with the newest "con" values
        std_data = scaler.transform([con_values])
        prediction = classifier.predict(std_data)
        # result = {
        #             0: "Buruk",
        #             1: "Kurang Subur",
        #             2: "Subur",
        #          }.get(prediction[0], "Sangat Subur")

        result = 'Buruk' if prediction[0] == 0 else ('Kurang Subur' if prediction[0] == 1 else ('Subur' if prediction[0] == 2 else 'Sangat Subur'))
        
        return jsonify({'forecasting': result, 'data_con': con_values})

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host='10.217.25.167',port=5000,debug=True)
