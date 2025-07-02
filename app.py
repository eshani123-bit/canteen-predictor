from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from io import TextIOWrapper, BytesIO
import csv
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ✅ MySQL Configuration
import os

# ✅ Secure and flexible DB configuration for Render/db4free
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# ✅ Load model
try:
    model = joblib.load('model.pkl')
except:
    model = LinearRegression()

# ✅ Mappings
menu_map = {'Normal': 0, 'Special': 1}
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

# ✅ User Table
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ✅ Prediction History Table
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    people = db.Column(db.Integer)
    menu = db.Column(db.String(50))
    event = db.Column(db.String(10))
    day = db.Column(db.String(20))
    predicted_kg = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="User already exists")
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    prediction = None

    if request.method == 'POST':
        people = int(request.form['people'])
        menu_text = request.form['menu']
        event_text = request.form['event']
        day_text = request.form['day']

        menu = menu_map[menu_text]
        event = 1 if event_text == 'Yes' else 0
        day = day_map[day_text]

        X_input = np.array([[people, menu, event, day]])
        pred = model.predict(X_input)[0]
        recommended = round(pred * 1.05, 2)
        prediction = f"Recommend preparing approximately {recommended} Kg of food."

        history = PredictionHistory(
            user_id=session['user_id'],
            people=people,
            menu=menu_text,
            event=event_text,
            day=day_text,
            predicted_kg=recommended
        )
        db.session.add(history)
        db.session.commit()

    all_history = PredictionHistory.query.filter_by(user_id=session['user_id']).order_by(PredictionHistory.timestamp.desc()).limit(7).all()
    line_data = [round(entry.predicted_kg, 2) for entry in reversed(all_history)]
    line_labels = [entry.timestamp.strftime("%b %d") for entry in reversed(all_history)]

    all_entries = PredictionHistory.query.filter_by(user_id=session['user_id']).all()
    day_data = dict(Counter([entry.day for entry in all_entries]))
    menu_data = dict(Counter([entry.menu for entry in all_entries]))

    return render_template(
        'dashboard.html',
        prediction=prediction,
        line_data=line_data,
        line_labels=line_labels,
        day_data=day_data,
        menu_data=menu_data
    )

@app.route('/retrain', methods=['POST'])
def retrain():
    file = request.files['file']
    if not file or file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))
    try:
        df = pd.read_csv(file)
        df['menu'] = df['menu'].map(menu_map)
        df['event'] = df['event'].map(lambda x: 1 if x.lower() == 'yes' else 0)
        df['day'] = df['day'].map(day_map)
        X = df[['people', 'menu', 'event', 'day']]
        y = df['actual_food']
        new_model = LinearRegression()
        new_model.fit(X, y)
        joblib.dump(new_model, 'model.pkl')
        global model
        model = new_model
        flash('Model retrained successfully!', 'success')
    except Exception as e:
        flash(f'Retraining failed: {e}', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
