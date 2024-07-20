from src.utils.file_utils import validate_image, image_to_numpy
from src.core.credit_card_processor import credit_card_detector, get_zones_coords
from src.core.ocr_service import get_ocr_service
from src.models.model import CreditCardData
from fastapi import FastAPI, File, UploadFile
from time import time
import uvicorn

app = FastAPI()

# Create an instance of OCRService
ocrService = get_ocr_service()

@app.get("/api/health", status_code=200)
def get_service_status() -> dict:
    return {"status": "Ok"}

@app.post("/api/service/credit-card", status_code=200)
async def credit_card_service(payment_network: str, file: UploadFile = File(...)):
    print("Enter to credit_card_service()")
    print(f"Image from request --- {file}")
    
    # Valid that file is an image
    image = validate_image(file=file)
    # Convert file to numpy array
    img_np = image_to_numpy(image=image)
    # Detect the credit card and payment network
    credit_card, payment_network = credit_card_detector(img=img_np)
    response = CreditCardData()
    if credit_card is not None and payment_network is not None:
        response.payment_network = payment_network
        ocrService.set_img(img=credit_card)
        ocrService.set_zones_coords(zones=get_zones_coords(payment_network))
        response = ocrService.extract(entity=response)
        response.obs = "Succesfull process!"
    else:
        response.obs = "Can not detected a credit card."

    return response
    

        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
