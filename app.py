import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask import redirect, url_for,session
import csv
from functools import wraps

# Defining Flask App
app = Flask(__name__)
app.secret_key = '123456'

nimgs = 50
# Admin credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = '123456'
logged_in = False

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    predictions = model.predict_proba(facearray)
    max_confidence = np.max(predictions)
    if max_confidence < 0.6:  # Adjust the threshold as needed
        return "Unknown"
    else:
        return model.predict(facearray)[0]



def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
from flask import request
import requests

# Your reCAPTCHA secret key
RECAPTCHA_SECRET_KEY = '6Ldb09spAAAAAOuaiKswR-y6JsSQ35F3x-ZpEPHD'

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the reCAPTCHA response from the form
        recaptcha_response = request.form['g-recaptcha-response']
        
        # Verify the reCAPTCHA response with Google's reCAPTCHA API
        verification_response = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': RECAPTCHA_SECRET_KEY,
                'response': recaptcha_response
            }
        ).json()
        
        # Check if the reCAPTCHA verification was successful
        if verification_response['success']:
            # Continue with the login process if the reCAPTCHA verification is successful
            username = request.form['username']
            password = request.form['password']
            if username == 'admin' and password == '123456':
                # Set the 'logged_in' session variable to True upon successful login
                session['logged_in'] = True
                return redirect(url_for('home'))
            else:
                # Render the login page with an error message if credentials are invalid
                return render_template('login.html', message='Invalid username or password. Please try again.')
        else:
            # Render the login page with an error message if the reCAPTCHA verification fails
            return render_template('login.html', message='reCAPTCHA verification failed. Please try again.')
    return render_template('login.html')


@app.route('/home')

def home():
    print("Inside home route")
    if 'logged_in' not in session:
        print("Redirecting to login route")
        return redirect(url_for('login'))
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
@app.route('/logout')

def logout():
    session.pop('logged_in', None)  # Remove the 'logged_in' session variable
    return redirect(url_for('login'))  # Redirect to the login page

from flask import jsonify

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        # Get the reCAPTCHA response from the form
        recaptcha_response = request.form.get('g-recaptcha-response')

        
        # Verify the reCAPTCHA response with Google's reCAPTCHA API
        verification_response = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': RECAPTCHA_SECRET_KEY,
                'response': recaptcha_response
            }
        ).json()
        
        # Check if the reCAPTCHA verification was successful
        if verification_response['success']:
            # Continue with the reset password process if the reCAPTCHA verification is successful
            secret_code = request.form.get('secret_code')
            if secret_code == 'Abdul Rafay':
                session['logged_in'] = True
                return redirect(url_for('home'))
            else:
                # Optionally, you can render an error message if the secret code is incorrect
                error_message = 'Invalid secret code. Please try again.'
                return render_template('reset_password.html', error_message=error_message)
        else:
            # Render the reset password page with an error message if the reCAPTCHA verification fails
            error_message = 'reCAPTCHA verification failed. Please try again.'
            return render_template('reset_password.html', error_message=error_message)
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    last_detected_face_label = None  # Initialize last detected face label
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            face_label = identify_face(face.reshape(1, -1))
            if face_label == "Unknown":
                cv2.putText(frame, f'Unknown', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                last_detected_face_label = face_label  # Update last detected face label
                cv2.putText(frame, f'{face_label}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key
            break
    cap.release()
    cv2.destroyAllWindows()
    
    # Mark attendance only for the last detected face
    if last_detected_face_label:
        add_attendance(last_detected_face_label)
    
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
            if i == 100:  # Capture 30 images
                break
        if i == 100:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

from flask import flash
'''@app.route('/feed_backs', methods=['POST'])
def feedback():
    pass'''
@app.route('/feeder')
def feeder():
    return render_template('feeder.html')

@app.route('/feed_backs', methods=['POST'])
def feedback():
    name = request.form['name']
    email = request.form['email']
    feedback = request.form['feedback']
    
    # Save feedback to a CSV file
    with open('feedback.csv', 'a', newline='') as csvfile:
        fieldnames = ['Name', 'Email', 'Feedback']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writerow({'Name': name, 'Email': email, 'Feedback': feedback})
    
    # Redirect to feeder page after feedback submission
    return redirect(url_for('feeder'))




@app.route('/manual-attendance', methods=['POST'])
def manual_attendance():
    if request.method == 'POST':
        manual_username = request.form['manualusername']
        manual_userid = request.form['manualuserid']
        current_time = datetime.now().strftime("%H:%M:%S")
        datetoday = date.today().strftime("%m_%d_%y")  # Get today's date
        
        # Check if the user exists in the database
        user_list = os.listdir('static/faces')
        user_exists = False
        for user in user_list:
            if f"{manual_username}_{manual_userid}" in user:
                user_exists = True
                break
        
        if user_exists:
            # Check if attendance for this user has already been marked today
            with open(f'Attendance/Attendance-{datetoday}.csv', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if f"{manual_username},{manual_userid}" in line:
                        error_message = 'Attendance for this user has already been marked today.'
                        return render_template('home.html', error=error_message)
            
            # Append manual attendance to CSV file
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{manual_username},{manual_userid},{current_time}')
            
            # Redirect to home page after submitting manual attendance
            return redirect(url_for('home'))
        else:
            # Render the login page with an error message
            error_message = 'The entered user does not exist in the database. Please check the name and ID.'
            return render_template('home.html', error=error_message)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        # Retrieve form data
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']

        # Write form data to CSV file
         # Write form data to a CSV file
    with open('contacts.csv', 'a', newline='') as csvfile:
        fieldnames = ['First Name', 'Last Name', 'Email', 'Phone', 'Message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the CSV file is empty and write headers if necessary
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write form data to CSV
        writer.writerow({'First Name': first_name, 'Last Name': last_name, 'Email': email, 'Phone': phone, 'Message': message})

    return redirect('/contact?success=true')

if __name__ == '__main__':
    app.run(debug=True)
