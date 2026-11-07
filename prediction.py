from datetime import datetime
import io
import os
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, make_response, redirect, render_template, request, session, url_for
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASEDIR = Path(__file__).resolve().parent
MODEL_PATH = BASEDIR / 'heart_disease_model.pkl'
SCALER_PATH = BASEDIR / 'scaler.pkl'

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-in-production')
app.config['SESSION_COOKIE_SECURE'] = False

MODEL_ERROR = None
MODEL = None
SCALER = None


def load_model_artifacts():
    global MODEL, SCALER, MODEL_ERROR
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'Model file not found: {MODEL_PATH.name}')
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f'Scaler file not found: {SCALER_PATH.name}')

        with MODEL_PATH.open('rb') as model_file:
            MODEL = pickle.load(model_file)
        with SCALER_PATH.open('rb') as scaler_file:
            SCALER = pickle.load(scaler_file)
        MODEL_ERROR = None
    except Exception as exc:  # pragma: no cover - runtime dependency check
        MODEL = None
        SCALER = None
        MODEL_ERROR = str(exc)


load_model_artifacts()


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
    return render_template('main.html', error=MODEL_ERROR)


@app.route('/article')
def article():
    return render_template('article.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if MODEL_ERROR:
            return render_template('main.html', error=f'Model not ready: {MODEL_ERROR}. Train the model first.')

        user_data = {key: request.form[key] for key in request.form}

        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        invalid_fields = [field for field in required_fields if not user_data.get(field) or user_data.get(field).strip() == '' or user_data.get(field) == '----select option----']
        if invalid_fields:
            return render_template('main.html', error='Please complete all fields and select valid options before submitting.')

        try:
            data = np.array([[
                int(user_data['age']),
                int(user_data['sex']),
                int(user_data['cp']),
                int(user_data['trestbps']),
                int(user_data['chol']),
                int(user_data['fbs']),
                int(user_data['restecg']),
                int(user_data['thalach']),
                int(user_data['exang']),
                float(user_data['oldpeak']),
                int(user_data['slope']),
                int(user_data['ca']),
                int(user_data['thal'])
            ]])
        except ValueError:
            return render_template('main.html', error='Please enter valid numeric values and choose real options.')

        data_scaled = SCALER.transform(data)
        my_prediction = MODEL.predict(data_scaled)[0]
        probability = MODEL.predict_proba(data_scaled)[0][1]

        risk_percentage = round(probability * 100, 2)
        prediction_text = 'High Risk' if my_prediction == 1 else 'Low Risk'

        session['user_data'] = user_data
        session['prediction'] = prediction_text
        session['risk_percentage'] = risk_percentage

        return render_template('result.html', prediction=prediction_text, risk_percentage=risk_percentage)


@app.route('/download_report')
def download_report():
    user_data = session.get('user_data')
    prediction = session.get('prediction', 'Unknown')
    risk_percentage = session.get('risk_percentage', 0)

    if not user_data:
        return redirect(url_for('prediction'))

    pdf_buffer = generate_pdf(user_data, prediction, risk_percentage)
    return send_pdf_buffer(pdf_buffer, 'Heart_Disease_Report.pdf')


def send_pdf_buffer(pdf_buffer, filename):
    pdf_buffer.seek(0)
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Cache-Control'] = 'no-cache'
    return response


def generate_pdf(user_data, prediction, risk_percentage):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title='Early Detection of Heart Disease Through Predictive Machine Learning Models Report')
    elements = []
    styles = getSampleStyleSheet()

    title = Paragraph('<b><font size=18 color=red>Early Detection of Heart Disease Through Predictive Machine Learning Models Report</font></b><br/><hr/>', styles['Title'])
    elements.append(title)

    summary_data = [['Prediction', 'Value'], ['Risk Level', prediction], ['Risk Percentage', f'{risk_percentage}%']]
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

    user_data_list = [['Features', 'Values']] + [[key, value] for key, value in user_data.items()]
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
    diet_title = Paragraph('<b><font size=14 color=darkgreen>Diet & Health Suggestions</font></b>', styles['Heading2'])
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

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    footer = Paragraph(f'<font size=10 color=grey>Report Generated on: {timestamp}</font>', styles['Normal'])
    elements.append(footer)

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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)


