from src.core.credit_card_processor import credit_card_detector, get_zones_coords
from src.core.ocr_service import get_ocr_service
from src.models.model import CreditCardData
import gradio as gr
import cv2
import json

choices = ["VISA", "MASTER CARD", "AMERICAN EXPRESS", "CABAL"]

ocr_service = get_ocr_service()

def process(image_input, payment_network_input):
    print("Enter to process()")
    
    credit_card, payment_network = credit_card_detector(img=image_input)
    response = CreditCardData()
    if credit_card is not None and payment_network is not None:
        response.payment_network = payment_network
        ocr_service.set_img(img=credit_card)
        ocr_service.set_zones_coords(zones=get_zones_coords(payment_network))
        response = ocr_service.extract(entity=response)
        response.obs = "Succesfull process"
    else:
        response.abs = "Can not detected a credit card"

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
        payment_network_input = gr.Dropdown(choices=choices, label="Cases")

    with gr.Row():
        process_button = gr.Button("SUBMIT")
        
    with gr.Row():
        output_json = gr.JSON()
    
    process_button.click(process, 
                         inputs=[image_input, payment_network_input],
                         outputs=[output_json])
        
demo.launch(share=True)