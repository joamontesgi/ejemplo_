from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carga el modelo de Orange
model = joblib.load('mlp.pkcls')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data['feature1'], data['feature2'], data['feature3']]
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
