import io
from fastapi.testclient import TestClient
from fastapi_main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello. Go to http://127.0.0.1:8000/docs"}

def test_read_item():
    response = client.get("/User/testuser")
    assert response.status_code == 200
    assert "Hello testuser" in response.json()["message"]

def test_prediction_endpoint():
    with open("tests/sample1.jpg", "rb") as image_file:
        image_data = image_file.read()


    response = client.post("/prediction/?probability=0.5", files={"file": ("filename", io.BytesIO(image_data), "image/jpeg")})
    assert response.json()["Summ"] == 300
    assert response.status_code == 200
    assert "filename" in response.json()
    assert "Summ" in response.json()
