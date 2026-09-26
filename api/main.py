from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()


# Load trained ML model
model = joblib.load("models/student_model.pkl")


# Input data structure
class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_grade: float
    assignments: int


# Home endpoint
@app.get("/")
def home():
    return {"message": "Student Performance ML API"}


# Prediction endpoint
@app.post("/predict")
def predict(student: StudentData):

    features = [[
        student.study_hours,
        student.attendance,
        student.previous_grade,
        student.assignments
    ]]

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    return {
        "prediction": int(prediction),
        "result": result
    }