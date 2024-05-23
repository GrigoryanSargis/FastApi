import io
from PIL import Image
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Path
from fastapi.responses import JSONResponse, HTMLResponse
import datetime
from urllib.parse import unquote, quote
import base64

from model import model



app = FastAPI(
    title="Coin recognition API",
    description="Processes uploaded images and returns sum of coins.",
    version="6.6.6",
    openapi_tags=[
                {"name": "Greeting", "description": "API endpoints for greeting users"},
        {"name": "Model Info", "description": "API endpoint related to model information"},
        {"name": "Prediction", "description": "API endpoints related to image prediction"},

    ]
)

@app.get("/", tags=["Greeting"])
def root():
    """
    Greet a user.

    This endpoint returns a greeting message.
    """
    return {"message": "Hello. Go to http://127.0.0.1:8000/docs"}


@app.get("/User/{name}", tags=["User"])
def User_name(name:str):
    """
    Greet a specific user.

    Args:
        name (str): The name of the user to greet.

    Returns:
        dict: A greeting message including the user's name and the current date.
    """
    return {"message": f"Hello {name}, today is {datetime.date.today()}"}


@app.get("/model-info/", tags=["Model Info"])
def get_model_info():
    """Provide details about the model in use."""
    model_info = {
        "algorithm_name": "YOLOv5",
        "related_research_paper": [
            "https://arxiv.org/abs/1506.02640"
        ],
        "version_number": "7.0",
        "training_dataset": "Custom dataset of coin images"
    }
    return model_info

@app.post("/Prediction/", tags=['Prediction'])
async def predict(    
    file: UploadFile = File(None, description="A required image file for prediction."),
    probability: float = Query(0.5, description="The confidence threshold for predictions.", ge=0, le=1)
):

    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    image_data = await file.read()
    try:
        image = Image.open(io.BytesIO(image_data))
    except IOError:
        return JSONResponse(status_code=400, content={"message": "Invalid image format."})

    model.conf = probability
    results = model(image)

    coin_sum = 0
    for q in range(len(results.pandas().xyxy[0])):
      coin_sum += int(results.pandas().xyxy[0]['name'][q])


    return {"filename": file.filename, "Summ": coin_sum}

@app.get("/PredictionFromURL/{url_path:path}", tags=["Prediction"])
async def predict_from_url(
    url_path: str = Path(..., description="The URL of the image to predict."),
    probability: float = Query(0.5, description="The confidence threshold for predictions.", ge=0, le=1)
):
    image_url = unquote(url_path)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image from URL: {str(e)}<n/>Please use /encode-url/ then paste here")

    try:
        image = Image.open(io.BytesIO(response.content))
    except IOError:
        return JSONResponse(status_code=400, content={"message": "Invalid image format."})
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")


    model.conf = probability
    results = model(image,size= 240)

    # Extract the sum of detected objects (assuming 'name' contains numeric values of the coins)
    coin_sum = 0
    for q in range(len(results.pandas().xyxy[0])):
        coin_sum += int(results.pandas().xyxy[0]['name'][q])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predicted Image</title>
    </head>
    <body>
        <h1>Predicted Image</h1>
        <p>Sum of coins: {coin_sum}</p>
        <img src="data:image/png;base64,{img_str}" alt="Predicted Image"/>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("fastapi_main:app", host="127.0.0.1", port=8000, reload=True)
