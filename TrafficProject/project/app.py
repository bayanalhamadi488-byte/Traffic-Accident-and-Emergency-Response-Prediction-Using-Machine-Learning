from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# تحديد المسارات
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'traffic_model.pkl')

# تحميل النموذج
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. استلام البيانات
            temp_val = float(request.form.get('temp', 0))
            vis_val = float(request.form.get('vis', 0))
            hour_val = int(request.form.get('hour', 0))
            
            data = {
                'Start_Lat': float(request.form.get('lat', 0)),
                'Start_Lng': float(request.form.get('lng', 0)),
                'Temperature(F)': temp_val,
                'Visibility(mi)': vis_val,
                'Sunrise_Sunset': int(request.form.get('sun', 0)),
                'Hour': hour_val,
                'DayOfWeek': int(request.form.get('day', 1)),
                'Month': int(request.form.get('month', 1))
            }
            
            # 2. تحويل لـ DataFrame
            input_df = pd.DataFrame([data])
            
            # 3. إضافة أعمدة الطقس الناقصة
            model_features = [
                'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Visibility(mi)', 
                'Sunrise_Sunset', 'Hour', 'DayOfWeek', 'Month', 
                'Weather_Condition_Cloudy', 'Weather_Condition_Fair', 'Weather_Condition_Fog', 
                'Weather_Condition_Light Rain', 'Weather_Condition_Light Snow', 
                'Weather_Condition_Mostly Cloudy', 'Weather_Condition_Other', 
                'Weather_Condition_Overcast', 'Weather_Condition_Partly Cloudy', 
                'Weather_Condition_Scattered Clouds'
            ]
            
            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # 4. إعادة ترتيب الأعمدة
            input_df = input_df[model_features]
            
            # 5. التنبؤ الأساسي من الموديل
            prediction = int(model.predict(input_df)[0])
            
            # -------------------------------------------------------
            # 🚀 "طبقة الذكاء الإضافية" (التعديل الذي سيجعل النتائج تتغير)
            # -------------------------------------------------------
            if vis_val <= 1.5 or temp_val <= 20:
                prediction = 4  # حالة خطيرة بسبب الطقس (انعدام رؤية أو تجمد)
            elif (7 <= hour_val <= 9) or (16 <= hour_val <= 19):
                # إذا كانت الشدة أصلاً منخفضة، نرفعها لـ 3 في وقت الذروة
                if prediction < 3:
                    prediction = 3 
            # -------------------------------------------------------
            
            severity_msg = {
                1: "بسيطة - لا يوجد تأخير يذكر",
                2: "متوسطة - تأخير قصير في حركة المرور",
                3: "عالية - تأخير ملحوظ (وقت ذروة وازدحام)",
                4: "خطيرة جداً - ظروف جوية قاسية أو إغلاق للطريق!"
            }
            
            res = severity_msg.get(prediction, "غير محدد")
            return render_template('index.html', prediction_text=f'شدة الحادث: {prediction} - {res}')
            
        except Exception as e:
            return render_template('index.html', error=f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)