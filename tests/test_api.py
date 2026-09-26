from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Student Performance ML API"
    }


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "study_hours": 6,
            "attendance": 85,
            "previous_grade": 72,
            "assignments": 8
        }
    )

    assert response.status_code == 201

    data = response.json()

    assert "prediction" in data
    assert "result" in data

    assert data["prediction"] in [0, 1]
    assert data["result"] in ["Pass", "Fail"]