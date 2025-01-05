# Stock Price Prediction Using Machine Learning (Django Backend)

This project uses an LSTM (Long Short-Term Memory) algorithm to predict the next day's stock price. Below are the detailed steps to set up and run this project locally.

## Steps to Set Up and Run the Project Locally

### 1. Clone the Repository

Clone the project repository from the source using the following command:
```bash
git clone <repository-url>
```

### 2. Open the Project in Visual Studio Code

Navigate to the project folder and open it in Visual Studio Code (or your preferred code editor).

### 3. Install Virtual Environment

Install the virtual environment package if it's not already installed:
```bash
pip install virtualenv
```

### 4. Set Up the Virtual Environment

Create and activate a virtual environment for the project:
```bash
python -m venv my_env

# On Windows
my_env\Scripts\activate

# On macOS/Linux
source my_env/bin/activate
```

### 5. Install Dependencies

While inside the virtual environment, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 6. Start the Development Server

Run the Django development server:
```bash
python manage.py runserver
```

After running the command, a link will appear in the terminal (e.g., `http://127.0.0.1:8000/`). Copy and paste this link into your browser to access the application.

### 7. You're Ready to Go!

The project should now be running locally on your system. You can explore its features and functionality from your browser.

---

**Note:**
- Ensure that Python (version 3.7 or higher) and pip are installed on your system.
- Update the `requirements.txt` file if additional dependencies are required for the project.

