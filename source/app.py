
import os, sys
from imports import *
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--ckp_path", type = str), parser.add_argument("--multilabel", action = "store_true")
parser.add_argument("--port", type = int)
args = parser.parse_args()

from apis.predict import Predictor
predictor = Predictor(args.ckp_path)

app = fastapi.FastAPI(title = "LightX3ECG")
@app.get("/")
async def home():
    return "LightX3ECG"

@app.get("/predict")
async def get_pred(ecg_file):
    pred = predictor.get_pred(
        ecg_file, 
        is_multilabel = args.multilabel, 
    )
    return pred

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host = "0.0.0.0", port = args.port, 
    )