import time
from fastapi import File, HTTPException, UploadFile, APIRouter, status
from fastapi.responses import JSONResponse
import logging
import sys
import psutil
from project.image_validator.service.image_validator_api import validate_image_for_upload

# Initialize the router
router = APIRouter()

# Set up logging
logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@router.post("/validate-uploaded-image/")
async def validate_uploaded_image(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    start_cpu = psutil.cpu_percent()
    start_memory = psutil.virtual_memory().used
    
    try:
        image_bytes = await file.read()
        
        is_valid, message = validate_image_for_upload(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        return JSONResponse(
            status_code=200,
            content={"message": "Image is valid for upload"}
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "An unexpected error occured", "error": str(e)},
        )
        
    finally:
        end_time = time.perf_counter()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().used
        
        logger.info(f"Processing time: {end_time - start_time:.2f} sec")
        logger.info(f"CPU usage: {end_cpu - start_cpu:.2f}%")
        logger.info(f"Memory usage: {(end_memory - start_memory) / (1024 * 1024):.2f} MB")