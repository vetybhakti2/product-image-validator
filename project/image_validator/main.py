from fastapi import FastAPI
from project.image_validator.route.main import (
    router as image_validator_router,
)

app = FastAPI(
    title="Image Validator API",
    version="1.0.0"
)

app.include_router(image_validator_router)