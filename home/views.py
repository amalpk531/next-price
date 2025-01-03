import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to 'Agg' for non-GUI usage
from django.shortcuts import render, HttpResponse
from datetime import datetime, timedelta
import yfinance as yf
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
#--add-b-my-------------------------- 
from django.core.mail import message
from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from .models import Contact
# below import is done for sending emails
from django.conf import settings
from django.core.mail import send_mail
from django.core import mail
from django.core.mail.message import EmailMessage
#-add-b-my-------------- 

# Load the model
model_path = os.path.join('static', 'latest_oneday.h5')
model = tf.keras.models.load_model(model_path)
#----------------------------------------------#
def remove_last_predicted_row():
    """Remove the last row if it was a prediction (has 0 volume)"""
    try:
        df = pd.read_csv('data.csv', skiprows=[1,2])  # Skip ticker and date rows
        if df.iloc[-1]['Volume'] == 0:  # Check if last row was a prediction
            df = df.iloc[:-1]  # Remove last row
            # Save back to CSV with original format
            with open('static\data.csv', 'w') as f:
                df.to_csv(f, index=False, header=False)
    except Exception as e:
        print(f"Error removing last prediction: {e}")

def append_prediction_to_csv(date, close, high, low, open_price):
    """Append predicted values to data.csv"""
    try:
        # First remove any existing prediction
        remove_last_predicted_row()
        
        # Format the new row
        new_row = f"{date},{close},{high},{low},{open_price},0\n"  # Volume set to 0 to mark as prediction
        
        # Append the new row to the file
        with open('static\data.csv', 'a') as f:
            f.write(new_row)
            
    except Exception as e:
        print(f"Error appending prediction: {e}")
#----------------------------------------------#
# Create your views here.
def home(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    if request.method=="POST":
        fname=request.POST.get("name")
        femail=request.POST.get("email")
        desc=request.POST.get("desc")
        query=Contact(name=fname,email=femail,description=desc)
        query.save()
        messages.success(request, "Thanks For Reaching Us! We will get back to you soon....")
        return redirect('/contact')
    return render(request,'contact.html')

def handlelogin(request):
    if request.method=="POST":
        uname=request.POST.get("username")
        pass1=request.POST.get("pass1")
        myuser=authenticate(username=uname,password=pass1)
        if myuser is not None:
            login(request,myuser)
            # messages.success(request,"Login Success")
            return redirect('/')
        else:
            messages.error(request,"Invalid Credentials")
            return render(request,'login.html')
    return render(request,'login.html')

def handlesignup(request):
    if request.method=="POST":
        uname=request.POST.get("username")
        email=request.POST.get("email")
        password=request.POST.get("pass1")
        confirmpassword=request.POST.get("pass2")
        # Password match check
        if password != confirmpassword:
            messages.warning(request,"Passwords do not match")
            return redirect('/signup')

        # Username and email uniqueness check
        if User.objects.filter(username=uname).exists():
            messages.info(request,"Username is already taken")
            return redirect('/signup')
        if User.objects.filter(email=email).exists():
            messages.info(request,"Email is already registered")
            return redirect('/signup')
    
        # Create user
        myuser = User.objects.create_user(uname, email, password)
        myuser.save()
        messages.success(request,"Signup successful! Please login.")
        return redirect('/login')
    
    return render(request,'signup.html')

def dashboard(request):
    return render(request, 'dashboard.html')

def handlelogout(request):
    logout(request)
    messages.info(request,"Logout Successful")
    return redirect('/login')

def fetch_data(stock_symbol):
    """Fetch historical stock data from Yahoo Finance."""
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = "2015-01-01"
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data.to_csv('static\data.csv')
    return data[['Open', 'High', 'Low', 'Close']]
    

# Get today's date
today = datetime.today()

# Add one day to today's date to get tomorrow's date
tomorrow = today + timedelta(days=1)

# Format tomorrow's date as YYYY-MM-DD
tomorrow_str = tomorrow.strftime('%Y-%m-%d')

def prepare_model_input(data, time_steps=60):
    """Prepare the last 60 days of data for model prediction."""
    # Normalize the data
    scaler = data.max()  # You might want to save/load your scaler
    scaled_data = data / scaler
    
    # Get the last 60 days
    last_60_days = scaled_data.tail(time_steps).values
    
    # Reshape for LSTM input [samples, time steps, features]
    model_input = np.array([last_60_days])
    return model_input, scaler

def prediction(request):
    if request.method == "POST":
        stock_symbol = request.POST.get('stock_symbol')

        try:
            # Fetch data for the stock symbol
            data = fetch_data(stock_symbol)
            
            # Prepare input for the model
            model_input, scaler = prepare_model_input(data)
            
            # Make prediction
            scaled_prediction = model.predict(model_input)
            
            # Inverse transform the prediction
            prediction = scaled_prediction[0] * scaler
            
            # Round predictions to 2 decimal places
            predicted_open = round(float(prediction.iloc[0]), 2)
            predicted_high = round(float(prediction.iloc[1]), 2)
            predicted_low = round(float(prediction.iloc[2]), 2)
            predicted_close = round(float(prediction.iloc[3]), 2)

            # Scatter plot for actual prices and predicted prices
            plt.figure(figsize=(12, 6))
            plt.plot(data.index[-30:], data[['Open', 'High', 'Low', 'Close']].iloc[-30:], label='Actual Prices', color='blue')

            # Ensure predicted_next_day is a 2D array and access each predicted value
            predicted_next_day = np.array([[predicted_open, predicted_high, predicted_low, predicted_close]])
            predicted_date = tomorrow  # Predicted date is tomorrow's date

            # Plot the predicted prices
            plt.scatter(predicted_date, predicted_open, color='red', label='Predicted Open')
            plt.scatter(predicted_date, predicted_high, color='green', label='Predicted High')
            plt.scatter(predicted_date, predicted_low, color='yellow', label='Predicted Low')
            plt.scatter(predicted_date, predicted_close, color='purple', label='Predicted Close')

            # Connect the last actual price to the predicted price
            last_actual_point = data.iloc[-1][['Open', 'High', 'Low', 'Close']].values  # Last actual prices
            plt.plot(
                [data.index[-1], predicted_date],
                [last_actual_point[0], predicted_open],
                color='red', linestyle='--', label='Open Connection'
            )
            plt.plot(
                [data.index[-1], predicted_date],
                [last_actual_point[1], predicted_high],
                color='green', linestyle='--', label='High Connection'
            )
            plt.plot(
                [data.index[-1], predicted_date],
                [last_actual_point[2], predicted_low],
                color='yellow', linestyle='--', label='Low Connection'
            )
            plt.plot(
                [data.index[-1], predicted_date],
                [last_actual_point[3], predicted_close],
                color='purple', linestyle='--', label='Close Connection'
            )
            #----------------------------------------------#
            # Append prediction to CSV
            append_prediction_to_csv(
                tomorrow_str,
                predicted_close,
                predicted_high,
                predicted_low,
                predicted_open
            )

            #----------------------------------------------#
            # Add labels, legend, and title
            plt.title(f'{stock_symbol} Stock Price Prediction (Next Day: {tomorrow_str})')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # Save plot to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            buffer.close()

            # Prepare context with predicted values and plot
            context = {
                "predicted_open": predicted_open,
                "predicted_high": predicted_high,
                "predicted_low": predicted_low,
                "predicted_close": predicted_close,
                "graph": graph_base64,
                "symbol": stock_symbol,
                "t_date": tomorrow_str
            }

            return render(request, 'prediction.html', context)

        except Exception as e:
            error_message = f"Error processing {stock_symbol}: {str(e)}"
            return render(request, 'prediction.html', {"error": error_message})

    return render(request, 'prediction.html')
