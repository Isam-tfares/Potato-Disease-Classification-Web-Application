from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
# import joblib
from tensorflow import keras
from io import BytesIO
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@csrf_exempt
def predict_image(request):
    prediction=[0,0,0]
    model_path = 'C:\\Users\\hp\\Desktop\\DS_Parcours\\deep_learning_codebasics\\Potato Disease Classification\\saved_model\\my_model.h5'
    loaded_model = keras.models.load_model(model_path)
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image from the request
        image = request.FILES['image']
        
        # Read the image using BytesIO and decode with OpenCV
        image_data = BytesIO(image.read())
        img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            
            # Get the dimensions of the processed image
            height, width, channels = img.shape

            img=img.reshape((1, 128, 128, 3))
            prediction = loaded_model.predict(img)
            index=np.argmax(prediction)
            result = {'prediction': str(index)}
            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'Failed to read the image'})
    else:
        return JsonResponse({'error': 'Image file not found or method not allowed'})


