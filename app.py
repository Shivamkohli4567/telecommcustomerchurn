from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open('shubham.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

# Tips for Churn and Retention
churn_tips_data = [
    "Identify the Reasons: Understand why customers or employees are leaving...",
    "Improve Communication: Maintain open and transparent communication channels...",
    # ... Add more tips as needed
]

retention_tips_data = [
    "Provide Exceptional Customer Service...",
    "Create Loyalty Programs...",
    # ... Add more tips as needed
]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        return predict()
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    def convert_input(value):
        if value is None or str(value).lower() == 'nan':
            return -1  # Represent missing values with -1
        elif value and str(value).lower() == 'yes':
            return 1
        else:
            return 0

    gender = request.form.get("gender")
    gender = 0 if gender == "Male" else 1 if gender == "Female" else -1

    SeniorCitizen = convert_input(request.form.get("SeniorCitizen"))
    Partner = convert_input(request.form.get("Partner"))
    Dependents = convert_input(request.form.get("Dependents"))
    
    try:
        TotalCharges = float(request.form.get("TotalCharges"))
    except (ValueError, TypeError):
        TotalCharges = -1  # Missing value

    tenure = int(request.form.get("Tenure") or -1)
    PhoneService = convert_input(request.form.get("PhoneService"))
    
    MultipleLines = request.form.get("MultipleLines")
    MultipleLines = 0 if MultipleLines == "No" else 1 if MultipleLines == "Yes" else -1
        
    Contract = request.form.get("Contract")
    if Contract == "Month-to-month":
        Contract = 1
    elif Contract == "One year":
        Contract = 2
    elif Contract == "Two year":
        Contract = 3
    else:
        Contract = -1

    features = np.array([gender, SeniorCitizen, Partner, Dependents, TotalCharges, tenure, PhoneService, MultipleLines, Contract])
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction == 1:
        result = "The Customer Is Likely To Churn."
        tips_data = churn_tips_data
        tips_df = 'Churn Tips'
    else:
        result = "The Customer Is Not Likely To Churn."
        tips_data = retention_tips_data
        tips_df = 'Retention Tips'

    return render_template('index.html', result=result, tips_data=tips_data, tips_df=tips_df)

if __name__ == "__main__":
    app.run(debug=True, port=2000)

