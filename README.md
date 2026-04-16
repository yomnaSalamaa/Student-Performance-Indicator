# 🎓 Student Performance Indicator

A Machine Learning web application that predicts a student's **Math score** based on demographic and academic information.
The project demonstrates an **end-to-end ML pipeline** with training, preprocessing, and deployment using Flask.

---

## 🚀 Demo

Users enter:

* Gender
* Race / Ethnicity
* Parental level of education
* Lunch type
* Test preparation course
* Reading score
* Writing score

The system returns the **Predicted Math Score**.

---

## 🧠 Machine Learning Pipeline

This project follows a modular ML architecture:

* Data Ingestion
* Data Transformation
* Model Training
* Prediction Pipeline
* Flask Web Application

The trained model and preprocessing pipeline are saved and loaded during inference.

---

## 🛠️ Tech Stack

**Machine Learning**

* Scikit-learn
* Pandas
* NumPy

**Backend**

* Flask

**Frontend**

* HTML
* Bootstrap (UI Styling)

**Others**

* Pickle (Model Saving)
* Python

---

## 📁 Project Structure

```
Student-Performance-Indicator/
│
├── artifacts/              # Saved model & preprocessor
├── notebook/               # EDA & model training
├── src/
│   ├── components/         # Data ingestion, transformation, training
│   ├── pipeline/           # Prediction pipeline
│   ├── utils.py
│   ├── exception.py
│   └── logger.py
│
├── templates/              # HTML files
│   ├── index.html
│   └── home.html
│
├── static/                 # Images / CSS
├── app.py                  # Flask app
├── requirements.txt
└── setup.py
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yomnaSalamaa/Student-Performance-Indicator.git
cd Student-Performance-Indicator
```

Create virtual environment (optional but recommended):

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
python app.py
```

Open your browser:

http://127.0.0.1:5000/predictdata

---

## 📊 Features

✅ End-to-End ML Pipeline
✅ Modular Project Structure
✅ Flask Web Interface
✅ Real-time Prediction
✅ Model Persistence
✅ Clean UI with Bootstrap

---

## 📈 Future Improvements

* Add model accuracy display
* Add visualization charts
* Deploy to cloud (Render / AWS)
* Add API endpoint
* Add multiple models comparison

---

## 👩‍💻 Author

**Yomna Salama**
Data Scientist

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
