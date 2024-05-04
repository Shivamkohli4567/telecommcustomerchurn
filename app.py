import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

# Load the pre-trained model
model = pickle.load(open('TelcoCustomerChurn.pkl', 'rb'))

app = Flask(__name__, template_folder="templates")
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Tips for Churn Prevention
churn_tips_data = {
   "TIPS FOR CUSTOMER PREVENTION": [
           "Identify the Reasons: Understand why customers or employees are leaving. Conduct surveys, interviews, or exit interviews to gather feedback and identify common issues or pain points.",
           "Improve Communication: Maintain open and transparent communication channels. Address concerns promptly and proactively. Make sure customers or employees feel heard and valued.",
           "Enhance Customer/Employee Experience: Focus on improving the overall experience. This could involve improving product/service quality or creating a more positive work environment for employees.",
           "Offer Incentives: Provide incentives or loyalty programs to retain customers. For employees, consider benefits, bonuses, or career development opportunities.",
           "Personalize Interactions: Tailor interactions and offers to individual needs and preferences. Personalization can make customers or employees feel more connected and valued.",
           "Monitor Engagement: Continuously track customer or employee engagement. For customers, this might involve monitoring product usage or website/app activity. For employees, assess job satisfaction and engagement levels.",
           "Predictive Analytics: Use data and predictive analytics to anticipate churn. Machine learning models can help identify patterns and predict which customers or employees are most likely to churn.",
           "Feedback Loop: Create a feedback loop for ongoing improvement. Regularly seek feedback, analyze it, and use it to make informed decisions and changes.",
           "Employee Training and Development: Invest in training and development programs for employees. Opportunities for growth and skill development can improve job satisfaction and loyalty.",
           "Competitive Analysis: Stay aware of what competitors are offering. Ensure your products, services, and workplace environment remain competitive in the market."
   ]
}

# Tips for Customer Retention (Not Churning)
retention_tips_data = {
   "TIPS FOR CUSTOMER RETENTION": [
           "Provide Exceptional Customer Service: Ensure that customers receive excellent customer service and support.",
           "Create Loyalty Programs: Reward loyal customers with discounts, special offers, or exclusive access to products/services.",
           "Regularly Communicate with Customers: Keep customers informed about updates, new features, and promotions.",
           "Offer High-Quality Products/Services: Consistently deliver high-quality products or services that meet customer needs.",
           "Resolve Issues Quickly: Address customer concerns and issues promptly to maintain their satisfaction.",
           "Build Strong Customer Relationships: Develop strong relationships with customers by understanding their needs and preferences.",
           "Provide Value: Offer value-added services or content that keeps customers engaged and interested.",
           "Simplify Processes: Make it easy for customers to do business with you. Simplify processes and reduce friction.",
           "Stay Responsive: Be responsive to customer inquiries and feedback, even on social media and review platforms.",
           "Show Appreciation: Express gratitude to loyal customers and acknowledge their continued support."
   ]
}

# Create DataFrames
churn_tips_df = pd.DataFrame(churn_tips_data)
retention_tips_df = pd.DataFrame(retention_tips_data)

@app.route("/", methods=["GET", "POST"])
def home():
   if request.method == 'POST':
       return predict()
   return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
   def convert_input(value):
       if value is None or str(value).lower() == 'none':
           return -1  # Represent missing values with -1
       elif value and str(value).lower() == 'yes':
           return 1
       else:
           return 0

   gender = request.form.get("gender")
   if gender and gender.lower() == "Male":
       gender = 0
   elif gender and gender.lower() == "Female":
       gender = 1
   else:
       gender = -1  # Missing value

   SeniorCitizen = convert_input(request.form.get("SeniorCitizen"))
   Partner = convert_input(request.form.get("Partner"))
   Dependents = convert_input(request.form.get("Dependents"))
   TotalCharges = float(request.form.get("Totalcharges") or -1)  # -1 for missing value
   tenure = int(request.form.get("Tenure") or -1)  # -1 for missing value
   PhoneService = convert_input(request.form.get("Phoneservice"))
   MultipleLines = request.form.get("MultipleLines")
   if MultipleLines and MultipleLines.lower() == "No":
       MultipleLines = 0
   elif MultipleLines and MultipleLines.lower() == 'Yes':
       MultipleLines = 1
   else:
       MultipleLines = -1  # Missing value
   Contract = request.form.get('Contract')
   if Contract and Contract.lower() == "Month-to-Month":
       Contract = 1
   elif Contract and Contract.lower() == "One year":
       Contract = 2
   elif Contract and Contract.lower() == "Two year":
       Contract = 3
   elif Contract and Contract.lower() == "No":
       Contract = 4
   else:
       Contract = -1  # Missing value

   features = np.array([gender, SeniorCitizen, Partner, Dependents, TotalCharges, tenure, PhoneService, MultipleLines, Contract])
   features = features.reshape(1, -1)

   # Handle missing values (replace with the mean or median of the training data)
   feature_means = np.array([0.5, 0.161, 0.666, 0.364, 64.761, 32.371, 0.637, 0.164, 2.23])  # Replace with actual means/medians
   features = np.where(features == -1, feature_means, features)

   # Scale the features
   ss = StandardScaler()
   features = ss.fit_transform(features)

   prediction = model.predict(features)

   if prediction[0] == 1:
       result = "The Customer Is Likely To Churn."
       tips_df = churn_tips_df
       tips_data = churn_tips_data
   else:
       result = "The Customer Is Unlikely To Churn And Is Still There."
       tips_df = retention_tips_df
       tips_data = retention_tips_data

   return render_template('index.html', result=result, tips_data=tips_data, tips_df=tips_df)

if __name__ == "__main__":
   app.run(debug=True, port=5000) 
