# %%
from src.core.card_service import get_card_service
from src.core.ocr_service import get_ocr_service
from src.models.model import CreditCardData
from src.config.config import Config
import cv2

IMG_PATH = "../data/input_data/credit-card-7.jpg"
img = cv2.imread(filename=IMG_PATH)

card_service = get_card_service()
ocr_service = get_ocr_service()

card = card_service.get_card_bbox(input_img=img, show=True)

card_elements = card_service.get_card_elements(card=card, show=True)

payment_network = card_service.classify_payment_network(element=card_elements['payment_network'], card=card)
print(payment_network)

config = Config()

data = CreditCardData()
data.payment_network = payment_network
ocr_service.set_img(img=card)
ocr_service.set_images(elements=card_elements)
ocr_service.set_zones_coords(zones=config.COMMON_CARD_ZONES)
data = ocr_service.extract(entity=data)

print("HOLA")
print(data.payment_network)
print(data.card_number)
print(data.expiry_date)
print(data.cardholder)

