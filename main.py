

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create the student dataset
data = {
    "Student_ID": list(range(1, 31)),

    "Attendance": [
        85, 60, 90, 75, 95, 55, 80, 65, 88, 70,
        92, 58, 78, 82, 68, 96, 62, 86, 73, 89,
        67, 91, 76, 59, 84, 71, 94, 64, 87, 79
    ],

    "Study_Hours": [
        3, 1, 4, 2, 5, 1, 3, 2, 4, 2,
        5, 1, 3, 4, 2, 5, 1, 4, 3, 4,
        2, 5, 3, 1, 4, 2, 5, 2, 4, 3
    ],

    "Previous_Marks": [
        72, 45, 80, 65, 88, 40, 70, 55, 78, 60,
        85, 42, 68, 75, 58, 90, 50, 76, 63, 82,
        54, 84, 66, 43, 74, 61, 87, 52, 79, 69
    ],

    "Assignments_Completed": [
        90, 50, 95, 70, 100, 45, 80, 60, 90, 75,
        95, 50, 85, 80, 65, 100, 55, 88, 70, 92,
        60, 96, 78, 48, 86, 72, 98, 58, 91, 82
    ],

    "Final_Marks": [
        78, 48, 86, 67, 92, 42, 74, 57, 83, 63,
        90, 44, 72, 79, 60, 95, 52, 81, 68, 85,
        56, 89, 71, 45, 80, 64, 93, 54, 84, 75
    ]
}

df = pd.DataFrame(data)

# 2. Select input features and target
X = df[
    [
        "Attendance",
        "Study_Hours",
        "Previous_Marks",
        "Assignments_Completed"
    ]
]

y = df["Final_Marks"]

# 3. Divide data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the Machine Learning model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict marks for test students
predictions = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MODEL TRAINED SUCCESSFULLY!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nACTUAL AND PREDICTED MARKS:")
results = pd.DataFrame({
    "Actual_Marks": y_test.values,
    "Predicted_Marks": predictions.round(2)
})
print(results)

print("\nMEAN ABSOLUTE ERROR (MAE):", round(mae, 2))
print("R-SQUARED SCORE:", round(r2, 3))

# 8. Predict marks for a new student
new_student = pd.DataFrame([{
    "Attendance": 85,
    "Study_Hours": 3,
    "Previous_Marks": 72,
    "Assignments_Completed": 90
}])

predicted_marks = model.predict(new_student)[0]

# Keep the displayed prediction within the marks range
predicted_marks = max(0, min(100, predicted_marks))

print("\nNEW STUDENT PREDICTION:")
print("Estimated Final Marks:", round(predicted_marks, 2), "/ 100")
