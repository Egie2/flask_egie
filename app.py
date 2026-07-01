from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Pastikan file model.pkl dan scaler.pkl ada di folder yang sama dengan app.py
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)  # Mengambil list model: [model_dt, model_svc]

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Ambil data input dari form HTML
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
        
        # 2. Ubah data menjadi DataFrame sesuai dengan urutan saat training
        df = pd.DataFrame([data])
        
        # 3. Lakukan normalisasi (scaling) menggunakan scaler yang sudah di-load
        X = scaler.transform(df)
        
        # 4. Ambil model berdasarkan pilihan user di dropdown/select form
        selected_model_name = request.form['model_choice']
        model_index = model_names.index(selected_model_name)
        clf = model[model_index]
        
        # 5. Lakukan prediksi
        y_pred = clf.predict(X)
        
        # 6. Tentukan hasil teks prediksi
        if y_pred[0] == 1:
            prediction = 'Diabetic (Positif Diabetes)'
        else:
            prediction = 'Not Diabetic (Negatif Diabetes)'
            
        return render_template('index.html', 
                               model_names=model_names, 
                               prediction_text=f'Hasil Prediksi ({selected_model_name}): {prediction}')

if __name__ == '__main__':
    app.run(debug=True)