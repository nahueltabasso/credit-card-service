from fastapi import FastAPI
from src.core.ocr_service import get_ocr_service
from src.core.card_service import get_card_service
from src.config.config import Config
from src.api.v1.routes import router as v1_router
from src.api.v2.routes import router as v2_router
import uvicorn

app = FastAPI()

# Register versioned routes
app.include_router(v1_router, prefix="/api/v1/service/credit-card", tags=["v1"])
app.include_router(v2_router, prefix="/api/v2/service/credit-card", tags=["v2"])

config = Config()

# Create an instance of OCRService
ocrService = get_ocr_service()
# Create an instance of CardService
card_service = get_card_service()

@app.get("/api/health", status_code=200)
def get_service_status() -> dict:
    return {"status": "Ok"}

# @app.post("/api/v1/service/credit-card", status_code=200)
# async def credit_card_service(payment_network: str, file: UploadFile = File(...)):
#     print("Enter to credit_card_service()")
#     print(f"Image from request --- {file}")
    
#     # Valid that file is an image
#     image = validate_image(file=file)
#     # Convert file to numpy array
#     img_np = image_to_numpy(image=image)
#     # Detect the credit card and payment network
#     credit_card, payment_network = credit_card_detector(img=img_np)
#     response = CreditCardData()
#     if credit_card is not None and payment_network is not None:
#         response.payment_network = payment_network
#         ocrService.set_img(img=credit_card)
#         ocrService.set_zones_coords(zones=get_zones_coords(payment_network))
#         response = ocrService.extract(entity=response)
#         response.obs = "Succesfull process!"
#     else:
#         response.obs = "Can not detected a credit card."

#     return response

# @app.post("/api/v2/service/credit-card", status_code=200)
# async def get_data(file: UploadFile = File(...)):
#     print("Enter to credit_card_service()")
#     print(f"Image from request --- {file}")
    
#     # Valid that file is an image
#     image = validate_image(file=file)
#     # Convert file to numpy array
#     img_np = image_to_numpy(image=image)
#     # Detect a card in the image
#     card = card_service.get_card_bbox(input_img=img_np)
#     response = CreditCardData()
#     if card is not None:
#         # Detect the card elements
#         card_elements = card_service.get_card_elements(card=card)
#         # Classify the payment network
#         payment_network = card_service.classify_payment_network(
#             element=card_elements
#         )
#         response.payment_network = payment_network
#         ocrService.set_img(img=card)
#         ocrService.set_images(elements=card_elements)
#         ocrService.set_zones_coords(zones=config.COMMON_CARD_ZONES)
#         response = ocrService.extract(entity=response)
#         response.obs = "Succesfull process!"
#     else:
#         response.obs = "Invalid image"
#     return response
    

        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
