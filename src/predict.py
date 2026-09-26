import joblib


# Load the trained model
model = joblib.load("models/student_model.pkl")


# A new student
student = [[6,25,72,8],
    [1, 0, 100, 8]]


# Make prediction
prediction = model.predict(student)


if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")