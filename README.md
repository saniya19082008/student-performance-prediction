# Student Performance Prediction Using Machine Learning

## Overview
This beginner project demonstrates how a machine-learning model can estimate a student's final marks using study hours per day, attendance percentage, and previous marks.

## Important limitation
The included dataset is **synthetic demonstration data**, not real student records. The predictions are for learning purposes only and must not be used for grading, admissions, or other real decisions.

## Technology
- Python
- Pandas
- Scikit-learn (Random Forest Regressor)
- Streamlit

## Run locally
1. Install Python 3.10 or newer.
2. Download/clone this repository.
3. Open a terminal in the project folder.
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`
6. Open the local URL shown by Streamlit.

## Files
- `app.py`: interactive web application and ML model
- `student_performance_data.csv`: small synthetic sample dataset
- `requirements.txt`: dependencies

## Method
1. Load the sample dataset.
2. Use study hours, attendance, and previous marks as input features.
3. Split the data into training and testing sets.
4. Train a Random Forest regression model.
5. display a predicted mark and basic evaluation metrics.

## Author
Saniya Nadar — edit this section if your program requires a different author format.
