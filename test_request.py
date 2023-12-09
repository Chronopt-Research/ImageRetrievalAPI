import requests
from PIL import Image
import io
import base64
import time

url = "http://localhost:80/reverseSearchImage/"
files = {"img": open("./test1.jpg","rb")}
response = requests.post(url, files=files)
img_dict = response.json()
# img_search = Image.open(io.BytesIO(response.content)).convert("RGB")
strart_time = time.time()
for key in img_dict.keys():
    img_bytes = base64.b64decode(img_dict[key])
    image = Image.open(io.BytesIO(img_bytes))
    # image.show()
print(time.time()-strart_time)
# img_search.show()
