from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# pastikan file ini ada di folder yang sama: model.pkl & scaler.pkl
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)  # diasumsikan: [Decision Tree, SVC]
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Pregnancies': int(request.form['pregnancies']),
        'Glucose': int(request.form['glucose']),
        'BloodPressure': int(request.form['blood_pressure']),
        'SkinThickness': int(request.form['skin_thickness']),
        'Insulin': int(request.form['insulin']),
        'BMI': float(request.form['bmi']),
        'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
        'Age': int(request.form['age'])
    }
    df = pd.DataFrame(data, index=[0])
    X = scaler.transform(df)

    clf = model[ model_names.index(request.form['model']) ]
    y = clf.predict(X)
    prediction = 'Diabetic' if int(y[0]) == 1 else 'Non-Diabetic'

    return render_template('index.html', model_names=model_names, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)