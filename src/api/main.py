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

# # Create an instance of OCRService
# ocrService = get_ocr_service()
# # Create an instance of CardService
# card_service = get_card_service()

@app.get("/api/health", status_code=200)
def get_service_status() -> dict:
    return {"status": "Ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
