from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/predict/")
async def predict(data: dict, token: str = Depends(oauth2_scheme)):
    # Here you would validate the token before proceeding
    input_df = pd.DataFrame(data)
    predictions = model.predict(input_df)
    return {"predictions": predictions.tolist()}
