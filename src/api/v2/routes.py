from fastapi import APIRouter, File, UploadFile
from src.utils.file_utils import validate_image, image_to_numpy
from src.core.ocr_service import get_ocr_service
from src.core.card_service import get_card_service
from src.config.config import Config
from src.models.model import CreditCardData

router = APIRouter()

config = Config()
ocrService = get_ocr_service()
card_service = get_card_service()

@router.post("/", status_code=200)
async def get_data(file: UploadFile = File(...)):
    print("Enter to get_data()")
    print(f"Image from request --- {file}")
    
    # Valid that file is an image
    image = validate_image(file=file)
    # Convert file to numpy array
    img_np = image_to_numpy(image=image)
    # Detect a card in the image
    card = card_service.get_card_bbox(input_img=img_np)
    response = CreditCardData()
    if card is not None:
        # Detect the card elements
        card_elements = card_service.get_card_elements(card=card)
        # Classify the payment network
        payment_network = card_service.classify_payment_network(
            element=card_elements['payment_network'],
            card=card
        )
        response.payment_network = payment_network
        ocrService.set_img(img=card)
        ocrService.set_images(elements=card_elements)
        ocrService.set_zones_coords(zones=config.COMMON_CARD_ZONES)
        response = ocrService.extract(entity=response)
        response.obs = "Succesfull process!"
    else:
        response.obs = "Invalid image"
    return response