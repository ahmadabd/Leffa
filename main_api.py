import os
import io
import base64
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from typing import Optional

# Import your LeffaPredictor from your app.py (assuming it is in the same directory)
# Either adjust the module path or copy the class definition to this file.
from app import LeffaPredictor

app = FastAPI(title="Leffa API", description="FastAPI endpoints for the Leffa model", version="1.0")

# Global predictor instance
leffa_predictor = LeffaPredictor()

def np_to_base64(np_img: np.ndarray) -> str:
    """Convert a NumPy image array to a base64 encoded PNG image."""
    pil_image = Image.fromarray(np.uint8(np_img))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """Save an uploaded file to a temporary location and return the file path."""
    try:
        suffix = os.path.splitext(upload_file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
            tmp_path = tmp.name
    finally:
        upload_file.file.close()
    return tmp_path

@app.post("/predict/vt")
async def predict_vt(
    src_image: UploadFile = File(..., description="Person image"),
    ref_image: UploadFile = File(..., description="Garment image"),
    ref_acceleration: bool = Form(False),
    step: int = Form(50, description="Inference steps"),
    scale: float = Form(2.5, description="Guidance scale"),
    seed: int = Form(42, description="Random seed"),
    vt_model_type: str = Form("viton_hd", description="Virtual try-on model type (viton_hd or dress_code)"),
    vt_garment_type: str = Form("upper_body", description="Garment type"),
    vt_repaint: bool = Form(False, description="Repaint mode"),
    preprocess_garment: bool = Form(False, description="Preprocess garment (for PNG files only)")
):
    """
    Endpoint for virtual try-on prediction.
    """
    # Save the uploaded files to temporary paths
    try:
        src_image_path = save_upload_file_tmp(src_image)
        ref_image_path = save_upload_file_tmp(ref_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to save uploaded images: " + str(e))
    
    try:
        # Run the model prediction for virtual try-on.
        generated_image, mask, densepose = leffa_predictor.leffa_predict_vt(
            src_image_path, ref_image_path, ref_acceleration,
            step, scale, seed, vt_model_type,
            vt_garment_type, vt_repaint, preprocess_garment
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during model inference: " + str(e))
    finally:
        # Clean up temporary files.
        os.remove(src_image_path)
        os.remove(ref_image_path)

    # Convert the resulting NumPy images to base64 encoded strings so they can be returned in JSON.
    response = {
        "generated_image": np_to_base64(generated_image),
        "mask": np_to_base64(mask),
        "densepose": np_to_base64(densepose),
    }
    return JSONResponse(content=response)

@app.post("/predict/pt")
async def predict_pt(
    src_image: UploadFile = File(..., description="Target pose person image"),
    ref_image: UploadFile = File(..., description="Source person image"),
    ref_acceleration: bool = Form(False),
    step: int = Form(50, description="Inference steps"),
    scale: float = Form(2.5, description="Guidance scale"),
    seed: int = Form(42, description="Random seed")
):
    """
    Endpoint for pose transfer prediction.
    """
    try:
        src_image_path = save_upload_file_tmp(src_image)
        ref_image_path = save_upload_file_tmp(ref_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to save uploaded images: " + str(e))
    
    try:
        generated_image, mask, densepose = leffa_predictor.leffa_predict_pt(
            src_image_path, ref_image_path, ref_acceleration,
            step, scale, seed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during model inference: " + str(e))
    finally:
        os.remove(src_image_path)
        os.remove(ref_image_path)

    response = {
        "generated_image": np_to_base64(generated_image),
        "mask": np_to_base64(mask),
        "densepose": np_to_base64(densepose),
    }
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
