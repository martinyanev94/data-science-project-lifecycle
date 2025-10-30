from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model/model.pkl")

@app.post("/predict/")
async def predict(data: dict):
    input_df = pd.DataFrame(data)
    predictions = model.predict(input_df)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
