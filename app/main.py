from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse,FileResponse, StreamingResponse
import torch
# from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2

app = FastAPI()

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=1, init_features=32, pretrained=False)
model.load_state_dict(torch.load("unet_model_48.pth",map_location=torch.device('cpu')))
device = torch.device("cpu")
model.to(device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)  
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        print("Orignal Shape : ",image.shape)
        image = cv2.resize(image, (512, 512))
        print("Resized Shape : ",image.shape)
        image = image.astype(np.float32) / 255.0
        input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        print("Resized Shape : ",input_tensor.shape)
        with torch.no_grad():
            output = model(input_tensor)
            mask = output.squeeze().cpu().numpy()

        mask = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask)  
        
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
