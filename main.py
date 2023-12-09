from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from utils.VectorDatabase import CosineSimVecDB
import cv2
from PIL import Image
import io
import secrets
import os
import time
import base64

# tempory_folder_path = "./temp/"

# def generate_random_token():
#     token = secrets.token_hex(16)
#     return token

app = FastAPI()
inMemDatabase = CosineSimVecDB("./embed_data","/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/","cuda")

# inMemDatabase = CosineSimVecDB(r"D:\resFes\Vector_database","/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/","cpu")

@app.post("/reverseSearchImage/")
async def videoColorization(img: UploadFile):
    img_bytes = await img.read()
    # rand_file_name=generate_random_token()+".jpg"
    img_search = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    response  = {}
    resp_img_paths = inMemDatabase.get_nearest_img_path(img_search)
    # img_search.save(tempory_folder_path+"/"+rand_file_name)
    for i in len(resp_img_paths):
        # print(resp_img_paths[i])
        with open(resp_img_paths[i], "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            response[f"img{str(i)}"]=encoded_string
    return response
