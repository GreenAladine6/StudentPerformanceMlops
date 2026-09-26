import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Load the dataset
data = pd.read_csv("data/students.csv")


# 2. Separate features and target
X = data[
    [
        "study_hours",
        "attendance",
        "previous_grade",
        "assignments"
    ]
]

y = data["passed"]


# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Create the model
model = LogisticRegression()


# 5. Train the model
model.fit(X_train, y_train)


# 6. Make predictions
predictions = model.predict(X_test)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)


# 8. Save the trained model
joblib.dump(model, "models/student_model.pkl")

print("Model saved to models/student_model.pkl")