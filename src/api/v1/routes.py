from fastapi import APIRouter, File, UploadFile
from src.utils.file_utils import validate_image, image_to_numpy
from src.core.credit_card_processor import credit_card_detector, get_zones_coords
from src.core.ocr_service import get_ocr_service
from src.models.model import CreditCardData

router = APIRouter()

# Create an instance of OCRService
ocrService = get_ocr_service()

@router.post("/", status_code=200)
async def credit_card_service(payment_network: str, file: UploadFile = File(...)):
    image = validate_image(file=file)
    img_np = image_to_numpy(image=image)
    credit_card, payment_network = credit_card_detector(img=img_np)
    response = CreditCardData()
    if credit_card and payment_network:
        response.payment_network = payment_network
        ocrService.set_img(img=credit_card)
        ocrService.set_zones_coords(zones=get_zones_coords(payment_network))
        response = ocrService.extract(entity=response)
        response.obs = "Successful process!"
    else:
        response.obs = "Can't detect credit card."
    return response
