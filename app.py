from datetime import datetime
from flask import Flask, render_template, request, session, make_response, redirect, url_for
import os
import pickle
import numpy as np
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

basedir = os.path.abspath(os.path.dirname(__file__))

# Load the Random Forest Classifier model
filename = os.path.join(basedir, 'heart_disease_model.pkl')
model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open(os.path.join(basedir, 'scaler.pkl'), 'rb'))

app = Flask(__name__)
app.secret_key = "your_secret_key"  
# Required for session storage

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/trends')
def trends():
    return render_template('trends.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/prediction')
def prediction():
    return render_template('main.html')

@app.route('/article')
def article():
    return render_template('article.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        user_data = {key: request.form[key] for key in request.form}

        # Validate all required fields and reject placeholder values
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        invalid_fields = [field for field in required_fields if not user_data.get(field) or user_data.get(field).strip() == '' or user_data.get(field) == '----select option----']
        if invalid_fields:
            return render_template('main.html', error="Please complete all fields and select valid options before submitting.")

        try:
            # Convert input data for model prediction
            data = np.array([[int(user_data['age']), int(user_data['sex']), int(user_data['cp']), int(user_data['trestbps']),
                              int(user_data['chol']), int(user_data['fbs']), int(user_data['restecg']), int(user_data['thalach']),
                              int(user_data['exang']), float(user_data['oldpeak']), int(user_data['slope']), int(user_data['ca']),
                              int(user_data['thal'])]])
        except ValueError:
            return render_template('main.html', error="Please enter valid numeric values and choose real options.")

        # Scale the input data
        data_scaled = scaler.transform(data)

        # Predict class and probability
        my_prediction = model.predict(data_scaled)[0]  # 0 (low risk) or 1 (high risk)
        probability = model.predict_proba(data_scaled)[0][1]  # Probability of class 1 (high risk)
        
        
        # Convert probability to percentage
        risk_percentage = round(probability * 100, 2)
        prediction_text = "High Risk" if my_prediction == 1 else "Low Risk"
        
        # Store results in session for PDF download
        session['user_data'] = user_data
        session['prediction'] = prediction_text
        session['risk_percentage'] = risk_percentage
        
        return render_template('result.html', prediction=prediction_text, risk_percentage=risk_percentage)


@app.route('/download_report')
def download_report():
    user_data = session.get('user_data')
    prediction = session.get('prediction', "Unknown")
    risk_percentage = session.get('risk_percentage', 0)

    if not user_data:
        return redirect(url_for('prediction'))

    pdf_buffer = generate_pdf(user_data, prediction, risk_percentage)
    return send_pdf_buffer(pdf_buffer, "Heart_Disease_Report.pdf")


def send_pdf_buffer(pdf_buffer, filename):
    pdf_buffer.seek(0)
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Cache-Control'] = 'no-cache'
    return response

def generate_pdf(user_data, prediction, risk_percentage):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="Early Detection of Heart Disease Through Predictive Machine Learning Models Report")  
    
    # Set Title

    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("<b><font size=18 color=red>Early Detection of Heart Disease Through Predictive Machine Learning Models Report</font></b><br/><hr/>", styles['Title'])
    # title_table = Table(title)
    # title_table.setStyle(TableStyle([
    #     ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
    #     ('FONTSIZE', (0, 0), (-1, -1), 16),
    #     ('TEXTCOLOR', (0, 0), (-1, -1), colors.red),
    #     ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    # ]))

    elements.append(title)
    
    # Prediction and Risk Percentage

    summary_data = [["Prediction","Value"],["Risk Level", prediction], ["Risk Percentage", f"{risk_percentage}%"]]
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkcyan),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    
    elements.append(Table([[' ']]))  
    
    # Spacer
    
    # User Input Data
    user_data_list = [["Features","Values"]]+ [[key, value] for key, value in user_data.items()]
    user_table = Table(user_data_list, colWidths=[200, 200])
    user_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkcyan),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(user_table)
    elements.append(Spacer(1, 20))

    diet_recommendations = get_diet_recommendations(prediction, risk_percentage)
    diet_title = Paragraph("<b><font size=14 color=darkgreen>Diet & Health Suggestions</font></b>", styles['Heading2'])
    elements.append(diet_title)
    elements.append(Spacer(1, 10))

    recommendation_rows = [[Paragraph(item, styles['Normal'])] for item in diet_recommendations]
    recommendation_table = Table(recommendation_rows, colWidths=[400])
    recommendation_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(recommendation_table)
    elements.append(Spacer(1, 20))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = Paragraph(f"<font size=10 color=grey>Report Generated on: {timestamp}</font>", styles['Normal'])
    elements.append(footer)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def get_diet_recommendations(prediction, risk_percentage):
    if risk_percentage > 40:
        return [
            '========== RECOMMENDED FOODS FOR HEART HEALTH ==========',
            '• High-Fiber Foods: Whole grains, vegetables, fruits',
            '• Healthy Fats: Salmon, nuts, olive oil',
            '• Lean Proteins: Skinless poultry, legumes, low-fat dairy',
            '• Heart-Friendly Beverages: Green tea, water, fresh vegetable juices',
            '',
            '========== FOODS TO AVOID ==========',
            '• Saturated & Trans Fats: Fried foods, processed snacks, red meat',
            '• Excess Salt: Canned soups, processed meats',
            '• Sugary Foods: White bread, pastries, sugary drinks',
            '',
            '========== HEART-HEALTHY MEAL PLAN ==========',
            '• Breakfast: Oatmeal with flaxseeds and berries + green tea',
            '• Lunch: Grilled salmon with quinoa & steamed broccoli',
            '• Snack: Walnuts & a banana',
            '• Dinner: Grilled chicken with a mixed greens salad',
            '• Evening Drink: Warm turmeric milk',
            '',
            '========== LIFESTYLE TIPS FOR HEART HEALTH ==========',
            '• Exercise regularly (30-45 minutes/day)',
            '• Get enough sleep (7-8 hours per night)',
            '• Reduce stress (meditation, breathing exercises)',
            '• Avoid smoking & alcohol',
        ]
    else:
        return [
            '========== NUTRIENT-RICH FOODS ==========',
            '• Whole Grains: Brown rice, whole wheat bread, oats',
            '• Vegetables: Leafy greens, carrots, broccoli',
            '• Fruits: Apples, berries, bananas, citrus fruits',
            '',
            '========== HIGH-QUALITY PROTEINS ==========',
            '• Lean Meats: Chicken, turkey',
            '• Eggs & Dairy: Low-fat milk, Greek yogurt',
            '• Plant-Based Proteins: Lentils, beans, tofu',
            '',
            '========== HEART-HEALTHY MEAL PLAN ==========',
            '• Breakfast: Oatmeal with flaxseeds and berries + green tea',
            '• Lunch: Grilled salmon with quinoa & steamed broccoli',
            '• Snack: Walnuts & a banana',
            '• Dinner: Grilled chicken with a mixed greens salad',
            '• Evening Drink: Warm turmeric milk',
            '',
            '========== LIFESTYLE TIPS FOR HEART HEALTH ==========',
            '• Exercise regularly (30-45 minutes/day)',
            '• Get enough sleep (7-8 hours per night)',
            '• Reduce stress (meditation, breathing exercises)',
            '• Avoid smoking & alcohol',
        ]

if __name__ == '__main__':
    app.run(debug=True)