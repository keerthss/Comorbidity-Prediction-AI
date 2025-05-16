from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load models with absolute paths
diabetes_model = tf.keras.models.load_model(r'C:\Users\joshp\Downloads\kittyyyxstevemlptabnet\models\diabetes_mlp.h5')
heart_model = tf.keras.models.load_model(r'C:\Users\joshp\Downloads\kittyyyxstevemlptabnet\models\heart_mlp.h5')
parkinsons_model = tf.keras.models.load_model(r'C:\Users\joshp\Downloads\kittyyyxstevemlptabnet\models\parkinson_mlp.h5')
hypertension_model = tf.keras.models.load_model(r'C:\Users\joshp\Downloads\kittyyyxstevemlptabnet\models\hypertension_mlp.h5')
celiac_model = tf.keras.models.load_model(r'C:\Users\joshp\Downloads\kittyyyxstevemlptabnet\models\celiac_mlp.h5')
kidney_model = tf.keras.models.load_model(r'C:\Users\joshp\Downloads\kittyyyxstevemlptabnet\models\kidney_mlp.h5')
obesity_model = tf.keras.models.load_model(r'C:\Users\joshp\Downloads\kittyyyxstevemlptabnet\models\obesity_mlp.h5')

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Login page
@app.route('/login')
def login():
    return render_template('login.html')

# Signup page
@app.route('/signup')
def signup():
    return render_template('signup.html')

# Diabetes Prediction
@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    if request.method == "POST":
        input_data = np.array([[float(request.form["pregnancies"]), 
                                float(request.form["glucose"]), 
                                float(request.form["bloodpressure"]), 
                                float(request.form["skinthickness"]), 
                                float(request.form["insulin"]), 
                                float(request.form["bmi"]), 
                                float(request.form["dpf"]), 
                                float(request.form["age"])]])
        
        prediction = diabetes_model.predict(input_data)
        prediction = prediction[0][0]  # Extract single float
        result = "Diabetic" if prediction > 0.5 else "Not Diabetic"
        return render_template("result.html", disease="Diabetes", result=result)
    return render_template("diabetes.html")

# Heart Disease Prediction
@app.route("/heart", methods=["GET", "POST"])
def heart():
    if request.method == "POST":
        input_data = np.array([[float(request.form["age"]), 
                                int(request.form["sex"] == "Male"), 
                                float(request.form["cp"]), 
                                float(request.form["trestbps"]), 
                                float(request.form["chol"]), 
                                int(request.form["fbs"] == "Yes"), 
                                float(request.form["restecg"]), 
                                float(request.form["thalach"]), 
                                int(request.form["exang"] == "Yes"), 
                                float(request.form["oldpeak"]), 
                                float(request.form["slope"]), 
                                float(request.form["ca"]), 
                                float(request.form["thal"])]])
        
        prediction = heart_model.predict(input_data)
        prediction = prediction[0][0]
        result = "Heart Disease Detected" if prediction > 0.5 else "No Heart Disease"
        return render_template("result.html", disease="Heart Disease", result=result)
    return render_template("heart.html")

# Parkinson's Prediction
@app.route("/parkinsons", methods=["GET", "POST"])
def parkinsons():
    if request.method == "POST":
        input_data = np.array([[float(request.form["fo"]), 
                                float(request.form["fhi"]), 
                                float(request.form["flo"]), 
                                float(request.form["jitter_percent"]), 
                                float(request.form["rap"]), 
                                float(request.form["hnr"]), 
                                float(request.form["spread1"]), 
                                float(request.form["spread2"]), 
                                float(request.form["d2"]), 
                                float(request.form["ppe"])]])
        
        prediction = parkinsons_model.predict(input_data)
        prediction = prediction[0][0]
        result = "Parkinson's Detected" if prediction > 0.5 else "No Parkinson's"
        return render_template("result.html", disease="Parkinson's", result=result)
    return render_template("parkinsons.html")

# Hypertension Prediction
@app.route("/hypertension", methods=["GET", "POST"])
def hypertension():
    if request.method == "POST":
        input_data = np.array([[float(request.form["age"]), 
                                float(request.form["bmi"]), 
                                float(request.form["systolic_bp"]), 
                                float(request.form["diastolic_bp"])]])
        
        prediction = hypertension_model.predict(input_data)
        prediction = prediction[0][0]
        result = "High Hypertension Risk" if prediction > 0.5 else "Low Hypertension Risk"
        return render_template("result.html", disease="Hypertension", result=result)
    return render_template("hypertension.html")

# Celiac Disease Prediction
@app.route("/celiac", methods=["GET", "POST"])
def celiac():
    if request.method == "POST":
        input_data = np.array([[float(request.form["iga"]), 
                                float(request.form["igg"]), 
                                float(request.form["tTG"])]])
        
        prediction = celiac_model.predict(input_data)
        prediction = prediction[0][0]
        result = "Celiac Disease Detected" if prediction > 0.5 else "No Celiac Disease"
        return render_template("result.html", disease="Celiac Disease", result=result)
    return render_template("celiac.html")

# Kidney Disease Prediction
@app.route("/kidney", methods=["GET", "POST"])
def kidney():
    if request.method == "POST":
        input_data = np.array([[float(request.form["blood_urea"]), 
                                float(request.form["serum_creatinine"]), 
                                float(request.form["sodium"]), 
                                float(request.form["potassium"])]])
        
        prediction = kidney_model.predict(input_data)
        prediction = prediction[0][0]
        result = "Kidney Disease Detected" if prediction > 0.5 else "No Kidney Disease"
        return render_template("result.html", disease="Kidney Disease", result=result)
    return render_template("kidney.html")

# Obesity Prediction
@app.route("/obesity", methods=["GET", "POST"])
def obesity():
    if request.method == "POST":
        input_data = np.array([[float(request.form["bmi"]), 
                                float(request.form["waist_circumference"]), 
                                float(request.form["physical_activity"])]])
        
        prediction = obesity_model.predict(input_data)
        prediction = prediction[0][0]
        result = "Obese" if prediction > 0.5 else "Not Obese"
        return render_template("result.html", disease="Obesity", result=result)
    return render_template("obesity.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
