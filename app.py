
from datetime import datetime
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Welcome - Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #081426; color: #F1F5F9; }
[data-testid="stSidebar"] { background-color: #10213B; }
h1, h2, h3 { color: #60A5FA; }
div[data-testid="stMetric"] { background-color: #142943; padding: 18px; border-radius: 12px; border: 1px solid #294768; }
.stButton > button { background-color: #2563EB; color: white; border-radius: 8px; border: none; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

FEATURES = ["Attendance", "Study_Hours", "Previous_Marks", "Assignments_Completed"]
TARGET = "Final_Marks"
REQUIRED_COLUMNS = ["Student_ID", "Attendance", "Study_Hours", "Previous_Marks", "Assignments_Completed", "Final_Marks"]

@st.cache_data
def load_data(file_source):
    data = pd.read_csv(file_source)
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for col in FEATURES + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES + [TARGET]).copy()
    data = data[data["Attendance"].between(0, 100) & data["Study_Hours"].between(0, 24) & data["Previous_Marks"].between(0, 100) & data["Assignments_Completed"].between(0, 10) & data["Final_Marks"].between(0, 100)]
    if len(data) < 5:
        raise ValueError("At least 5 valid records required.")
    return data

# SIDEBAR
st.sidebar.title("🎓 Student Analytics")
st.sidebar.write("Machine Learning Project")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("📁 Upload Custom Dataset (CSV)", type=["csv"])

# --- NEW LOGIC: Welcome + Default File Support ---
data_source = None
source_label = ""

if uploaded_file is not None:
    data_source = uploaded_file
    source_label = uploaded_file.name
elif os.path.exists("student_performance_100_same_format.csv"):
    data_source = "student_performance_100_same_format.csv"
    source_label = "Default Dataset"

if data_source is None:
    # WELCOME SCREEN
    st.markdown("""
        <div style='background: linear-gradient(135deg, #10213B 0%, #1E3A8A 100%); padding: 50px 30px; border-radius: 20px; text-align: center; border: 1px solid #294768; box-shadow: 0 10px 25px rgba(0,0,0,0.3); margin-top: 20px;'>
            <h1 style='color: #60A5FA; font-size: 42px; font-weight: 800; margin-bottom: 10px;'>WELCOME TO STUDENT-PERFORMANCE-PREDICTION</h1>
            <p style='color: #94A3B8; font-size: 19px; margin-top: 0;'>An advanced ML app to analyze and predict student success. - By DIVYANSHU SHARMA</p>
            <hr style='border: 0; height: 1px; background: #294768; margin: 30px auto; width: 60%;'>
            <div style='display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 25px;'>
                <div style='background: rgba(20, 41, 67, 0.7); padding: 20px; border-radius: 12px; width: 260px; border: 1px solid #3B82F6;'><h3 style='color: #60A5FA;'>📂 Step 1</h3><p style='color: #CBD5E1; font-size: 14px;'>Upload your CSV from sidebar or use default dataset.</p></div>
                <div style='background: rgba(20, 41, 67, 0.7); padding: 20px; border-radius: 12px; width: 260px; border: 1px solid #3B82F6;'><h3 style='color: #60A5FA;'>📊 Step 2</h3><p style='color: #CBD5E1; font-size: 14px;'>Explore analytics and insights.</p></div>
                <div style='background: rgba(20, 41, 67, 0.7); padding: 20px; border-radius: 12px; width: 260px; border: 1px solid #3B82F6;'><h3 style='color: #60A5FA;'>🔮 Step 3</h3><p style='color: #CBD5E1; font-size: 14px;'>Predict final marks instantly.</p></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- REST OF YOUR CODE SAME AS BEFORE ---
try:
    df = load_data(data_source)
except Exception as error:
    st.error(f"Dataset validation error: {error}")
    st.stop()

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction", "Prediction History", "Data Analysis", "Model Evaluation"])
st.sidebar.divider()
st.sidebar.write(f"📂 Source: {source_label}")
st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write(f"Students loaded: {len(df)}")
st.sidebar.write(f"Predictions made: {len(st.session_state['prediction_history'])}")

if page == "Dashboard":
    st.title("🎓 Student Performance Dashboard")
    st.write("Analyze student data and predict academic performance using Machine Learning.")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Average Final Marks", f"{df[TARGET].mean():.2f}")
    col3.metric("Average Attendance", f"{df['Attendance'].mean():.1f}%")
    col4.metric("Average Study Hours", f"{df['Study_Hours'].mean():.1f}")
    st.subheader("Student Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Average Marks Overview")
    chart_data = pd.DataFrame({"Marks Type": ["Previous Marks", "Final Marks"], "Average Marks": [df["Previous_Marks"].mean(), df["Final_Marks"].mean()]})
    st.bar_chart(chart_data.set_index("Marks Type"))
    st.caption(f"Active Dataset Source: {source_label}")

elif page == "Prediction":
    st.title("🔮 Predict Student Performance")
    st.write("Enter student details to estimate final marks.")
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", value="STU101")
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        study_hours = st.slider("Study Hours Per Day", 0.0, 12.0, 4.0, 0.5)
    with col2:
        previous_marks = st.slider("Previous Marks", 0, 100, 70)
        assignments = st.slider("Assignments Completed (out of 10)", 0, 10, 7)
    if st.button("Predict Final Marks"):
        if not student_id.strip():
            st.warning("Please enter a Student ID.")
        else:
            student_data = pd.DataFrame([[attendance, study_hours, previous_marks, assignments]], columns=FEATURES)
            predicted_marks = float(np.clip(model.predict(student_data)[0], 0, 100))
            report = {"Student_ID": student_id.strip(), "Attendance": attendance, "Study_Hours": study_hours, "Previous_Marks": previous_marks, "Assignments_Completed": assignments, "Predicted_Final_Marks": round(predicted_marks, 2), "Prediction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.session_state["latest_prediction"] = report
            st.session_state["prediction_history"].append(report.copy())
            st.success("Prediction completed and saved to history!")
    if "latest_prediction" in st.session_state:
        report = st.session_state["latest_prediction"]
        st.divider()
        st.subheader("📋 Prediction Result")
        st.write(f"Student ID: **{report['Student_ID']}**")
        marks = report["Predicted_Final_Marks"]
        st.metric("Estimated Final Marks", f"{marks:.2f} / 100")
        st.progress(int(round(marks)))
        if marks >= 75: st.success("Performance Category: Excellent")
        elif marks >= 60: st.info("Performance Category: Good")
        elif marks >= 40: st.warning("Performance Category: Average")
        else: st.error("Performance Category: Needs Improvement")
        st.subheader("Prediction Report")
        report_df = pd.DataFrame([report])
        st.dataframe(report_df, use_container_width=True)
        csv_report = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 Download Prediction Report", data=csv_report, file_name=f"{report['Student_ID']}_prediction_report.csv", mime="text/csv")
        st.caption("Predicted marks are estimates, not guaranteed results.")

elif page == "Prediction History":
    st.title("📚 Prediction History")
    st.write("View all predictions made during the current session.")
    history = st.session_state["prediction_history"]
    if len(history) == 0:
        st.info("No predictions saved yet. Open the Prediction page and predict marks for a student first.")
    else:
        history_df = pd.DataFrame(history)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(history_df))
        col2.metric("Average Predicted Marks", f"{history_df['Predicted_Final_Marks'].mean():.2f}")
        col3.metric("Different Student IDs", history_df["Student_ID"].nunique())
        st.subheader("All Saved Predictions")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        history_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 Download Complete Prediction History", data=history_csv, file_name="prediction_history.csv", mime="text/csv")
        st.divider()
        st.subheader("Manage History")
        confirm_clear = st.checkbox("I want to delete all prediction history")
        if st.button("Clear Prediction History"):
            if confirm_clear:
                st.session_state["prediction_history"] = []
                if "latest_prediction" in st.session_state: del st.session_state["latest_prediction"]
                st.success("Prediction history cleared.")
                st.rerun()
            else:
                st.warning("Please tick the confirmation checkbox first.")

elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    st.subheader(f"Complete Dataset ({source_label})")
    st.dataframe(df, use_container_width=True)
    st.subheader("Statistical Summary")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)
    st.subheader("Attendance vs Final Marks")
    st.scatter_chart(df, x="Attendance", y="Final_Marks")
    st.subheader("Study Hours vs Final Marks")
    st.scatter_chart(df, x="Study_Hours", y="Final_Marks")
    st.subheader("Previous Marks vs Final Marks")
    st.scatter_chart(df, x="Previous_Marks", y="Final_Marks")
    st.subheader("Correlation Between Features")
    correlation = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns))); ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right"); ax.set_yticklabels(correlation.columns)
    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)): ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥 Download Dataset CSV", data=csv_data, file_name="student_performance_cleaned.csv", mime="text/csv")

elif page == "Model Evaluation":
    st.title("🧠 Model Evaluation")
    st.write("This section measures how well the model predicts marks on test data.")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}"); col2.metric("RMSE", f"{rmse:.2f}"); col3.metric("R² Score", f"{r2:.3f}")
    st.subheader("Meaning of the Metrics")
    st.markdown("- **MAE:** Average absolute prediction error\n- **RMSE:** Prediction error that gives more weight to large errors.\n- **R² Score:** Closer to 1 is better.")
    results = pd.DataFrame({"Actual Marks": y_test.to_numpy(), "Predicted Marks": predictions})
    st.subheader("Actual vs Predicted Marks")
    st.dataframe(results.round(2), use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, predictions); ax.plot([0, 100], [0, 100], linestyle="--")
    ax.set_xlabel("Actual Final Marks"); ax.set_ylabel("Predicted Final Marks"); ax.set_title("Actual vs Predicted Marks")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Regression Coefficients")
    coefficients = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_})
    st.dataframe(coefficients.round(3), use_container_width=True)
    st.caption("Coefficients show estimated relationships, not proof of causation.")

st.sidebar.divider()
st.sidebar.caption("MCA Project | Student Performance Prediction")
st.markdown("---")
st.markdown("""<div style='text-align: right; font-family: "Georgia", serif; font-style: italic; font-size: 12px; color: #94A3B8; letter-spacing: 0.5px; padding-bottom: 10px;'>— developed by <b>DIVYANSHU SHARMA</b></div>""", unsafe_allow_html=True)
