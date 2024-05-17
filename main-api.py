from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
# include CORS
from fastapi.middleware.cors import CORSMiddleware
 
# Create the FastAPI app
 
# Add the following code to the FastAPI app to enable CORS
 
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Load the model
model = load_model('final_cnn.h5')
 
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = load_img(io.BytesIO(contents), target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_proba = prediction[0][predicted_class] * 100
    rounded_proba = np.round(predicted_proba, decimals=2)
    str_prob = str(rounded_proba)+ "%"
 
    class_names = ['Organic', 'Recyclable']
 
    return JSONResponse(content={'class': class_names[predicted_class[0]], 'probability': str_prob})
    # return JSONResponse(content={'class': str(predicted_class[0])})
    # return JSONResponse(content={'class': predicted_class})
 