# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request, session, flash
from werkzeug.utils import secure_filename
from supportFile import predict
import pandas as pd
import os
import cv2
import sqlite3
import predict
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, IntegerField, SelectField
from wtforms.validators import DataRequired, Length, Email, Regexp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Database Initialization
DATABASE = "mydatabase.db"

def create_db():
    """Initialize the database with required tables if they do not exist."""
    with sqlite3.connect(DATABASE) as con:
        cursor = con.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Date TEXT,
                Name TEXT,
                Contact TEXT,
                Email TEXT,
                Password TEXT,
                Age INTEGER,
                Gender TEXT
            )
        """)
        con.commit()

# Call function to create DB table
create_db()

# Flask-WTF Form
class RegistrationForm(FlaskForm):
    name = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email ID', validators=[DataRequired(), Email(), Length(max=100)])
    num = StringField('Contact No', validators=[DataRequired(), Regexp(r'^[0-9]{10}$', message="Must be a 10-digit number")])
    age = IntegerField('Age', validators=[DataRequired()])
    gender = StringField('Gender', validators=[DataRequired(), Length(max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=80)])

@app.route('/', methods=['GET', 'POST'])
def landing():
    return render_template('home1.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    form = RegistrationForm()  # Here is form
    if form.validate_on_submit():  # Validation occurs
        num = form.num.data  # Change here
        name = form.name.data  # Change here
        email = form.email.data  # Change here
        password = form.password.data  # Change here
        age = form.age.data  # Change here
        gender = form.gender.data  # Change here

        # Debug: Print the received form data
        print(f"Received form data: {name}, {email}, {num}, {age}, {gender}, {password}")

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        try:
            # Connect to SQLite database
            con = sqlite3.connect('mydatabase.db')
            cursorObj = con.cursor()
            
            # Ensure table exists
            cursorObj.execute("CREATE TABLE IF NOT EXISTS Users (Date text, Name text, Contact text, Email text, password text, age text, gender text)")
            
            # Insert data into the database
            cursorObj.execute("INSERT INTO Users (Date, Name, Contact, Email, password, age, gender) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (dt_string, name, num, email, password, age, gender))
            con.commit()
            con.close()  # Ensure the connection is closed after the commit
            print("✅ Data inserted successfully")
            return redirect(url_for('login'))  # Redirect to login page after successful submission

        except sqlite3.Error as e:
            print(f"❌ Error inserting data into database: {e}")
            return render_template('input.html', form=form, error="There was an error saving your data. Please try again.")  # Show error

    return render_template('input.html', form=form)  # Render the form again if validation fails

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        with sqlite3.connect(DATABASE) as con:
            cursor = con.cursor()
            cursor.execute("SELECT Name FROM Users WHERE Name=? AND Password=?", (name, password))
            user = cursor.fetchone()

            if user:
                return redirect(url_for('home'))
            else:
                error = "Invalid Credentials. Please try again."

    return render_template('login.html', error=error)

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/info', methods=['GET', 'POST'])
def info():
    return render_template('info.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        if request.form['sub'] == 'Upload':
            savepath = 'upload/'
            photo = request.files['photo']
            filename = secure_filename(photo.filename)
            photo.save(os.path.join(savepath, filename))

            image = cv2.imread(os.path.join(savepath, filename))
            cv2.imwrite(os.path.join("static/images/", "test_image.jpg"), image)
            return render_template('image.html')

        elif request.form['sub'] == 'Test':
            target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/images/')
            namefile_ = 'test_image.jpg'
            destination = os.path.join(target, namefile_)
            fruit, result = predict.predict(destination)
            return render_template('image.html', result=result, fruit=fruit)

    return render_template('image.html')

@app.after_request
def add_header(response):
    """No caching at all for API endpoints."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
