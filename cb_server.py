from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier
import joblib

app = Flask(__name__)

# 加载模型和编码器
model = CatBoostClassifier()
model.load_model("security.cbm")
label_encoders = joblib.load("security_label_encoders.pkl")

def preprocess_data(json_data):
    if isinstance(json_data, dict):
        json_data = [json_data]
    
    df = pd.DataFrame(json_data)
    encoded_data = {}
    for column in df.columns:
        if column in label_encoders:
            if df[column].dtype == object:
                df[column + '_encoded'] = df[column].apply(lambda x: label_encoders[column].transform([x])[0] if x in label_encoders[column].classes_ else -1)
                encoded_data[column + '_encoded'] = df[column + '_encoded'].tolist()
            else:
                encoded_data[column] = df[column].tolist()
    return pd.DataFrame(encoded_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_example = preprocess_data(data)
    predictions = model.predict(X_example)
    return '', 200, {'X-Prediction': str(predictions[0])}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
