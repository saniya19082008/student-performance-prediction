import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="📚", layout="centered")
st.title("📚 Student Performance Prediction")
st.write("A beginner machine-learning demonstration that estimates final marks from study hours, attendance, and previous marks.")
st.info("This is an educational demo using a small synthetic sample dataset. It is not suitable for real academic decisions.")

@st.cache_data
def load_data():
    return pd.read_csv("student_performance_data.csv")

df = load_data()
X = df[["study_hours", "attendance_percent", "previous_marks"]]
y = df["final_marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.subheader("Enter student information")
study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75, 1)
previous = st.slider("Previous marks (%)", 0, 100, 65, 1)

if st.button("Predict final marks"):
    estimate = model.predict(pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percent": attendance,
        "previous_marks": previous
    }]))[0]
    st.success(f"Estimated final marks: {estimate:.1f}%")
    st.caption("This is only an estimate from demonstration data, not a guarantee.")

with st.expander("View sample data"):
    st.dataframe(df, use_container_width=True)
with st.expander("Model information"):
    st.write("Algorithm: Random Forest Regressor")
    st.write(f"Mean Absolute Error on the held-out demo split: {mae:.2f} marks")
    st.write(f"R² on the held-out demo split: {r2:.2f}")
    st.write("Metrics can vary with this small synthetic dataset. Do not interpret them as real-world performance.")
st.subheader("Feature importance")
importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
st.bar_chart(importance.set_index("Feature"))
