from src.core.ocr_service import get_ocr_service
from src.core.card_service import get_card_service
from src.config.config import Config
from src.models.model import CreditCardData
import gradio as gr
import cv2
import json

config = Config()
ocr_service = get_ocr_service()
card_service = get_card_service()

def process(image_input):
    print("Enter to process()")
    
    card = card_service.get_card_bbox(input_img=image_input)
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
        ocr_service.set_img(img=card)
        ocr_service.set_images(elements=card_elements)
        ocr_service.set_zones_coords(zones=config.COMMON_CARD_ZONES)
        response = ocr_service.extract(entity=response)
        response.obs = "Succesfull process!"
    else:
        response.obs = "Invalid image"
    response_dict = response.to_dict()
    json_response = json.dumps(response_dict, indent=4)
    return json_response

########################################
#           GRADIO INTERFACE           #
########################################

with gr.Blocks() as demo:
    
    gr.Markdown(
        """
        # POC - CREDIT CARD SERVICE INTERFACE
        """
    )
    
    with gr.Row():
        image_input = gr.inputs.Image(type="numpy", label="Credit Card")

    with gr.Row():
        process_button = gr.Button("SUBMIT")
        
    with gr.Row():
        output_json = gr.JSON()
    
    process_button.click(process, 
                         inputs=[image_input],
                         outputs=[output_json])
        
demo.launch(share=True)