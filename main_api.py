import os
import io
import base64
import tempfile
import numpy as np
import hmac
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import contextlib

# Import your LeffaPredictor from your app.py (assuming it is in the same directory)
# Either adjust the module path or copy the class definition to this file.
from app import LeffaPredictor

# Simple Security Configuration
API_KEY = os.getenv("LEFFA_API_KEY", "123456")

app = FastAPI(
    title="Leffa API", 
    description="Endpoints for the Tryon model", 
    version="1.0",
    docs_url=False,
    redoc_url=False,
    openapi_url=False
)

# CORS Configuration - restrict to your services only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Global predictor instance
leffa_predictor = LeffaPredictor()

# Simple Security Functions
def verify_api_key(api_key: str) -> bool:
    """Verify API key using secure comparison."""
    return hmac.compare_digest(api_key, API_KEY)

# Security Dependency
async def verify_authentication(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple API key verification."""
    if not verify_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Utility Functions
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

@contextlib.contextmanager
def temp_files(*upload_files):
    """Context manager to handle temporary file cleanup automatically."""
    temp_paths = []
    try:
        for upload_file in upload_files:
            temp_path = save_upload_file_tmp(upload_file)
            temp_paths.append(temp_path)
        yield temp_paths
    finally:
        # Clean up all temporary files
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_path}: {e}")

# Secured API Endpoints
@app.post("/predict/vt")
async def predict_vt(
    vt_src_image: UploadFile = File(..., description="Person image"),
    vt_ref_image: UploadFile = File(..., description="Garment image"),
    vt_ref_acceleration: bool = Form(False, description="Accelerate Reference UNet"),
    vt_step: int = Form(30, description="Inference steps", ge=30, le=100),
    vt_scale: float = Form(2.5, description="Guidance scale", ge=0.1, le=5.0),
    vt_seed: int = Form(42, description="Random seed", ge=-1, le=2147483647),
    vt_model_type: str = Form("viton_hd", description="Model type: viton_hd or dress_code"),
    vt_garment_type: str = Form("upper_body", description="Garment type: upper_body, lower_body, or dresses"),
    vt_repaint: bool = Form(False, description="Repaint mode"),
    preprocess_garment: bool = Form(False, description="Preprocess garment (for PNG files only)"),
    authenticated: bool = Depends(verify_authentication)
):
    """
    Secured endpoint for virtual try-on prediction.
    Compatible with Gradio interface parameters.
    """
    
    # Use context manager for automatic cleanup
    with temp_files(vt_src_image, vt_ref_image) as (src_image_path, ref_image_path):
        try:
            # Run the model prediction for virtual try-on.
            generated_image, mask, densepose = leffa_predictor.leffa_predict_vt(
                src_image_path, ref_image_path, vt_ref_acceleration,
                vt_step, vt_scale, vt_seed, vt_model_type,
                vt_garment_type, vt_repaint, preprocess_garment
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error during model inference: " + str(e))

        # Convert the resulting NumPy images to base64 encoded strings so they can be returned in JSON.
        response = {
            "generated_image": np_to_base64(generated_image),
            "mask": np_to_base64(mask),
            "densepose": np_to_base64(densepose),
        }
        return JSONResponse(content=response)

@app.post("/predict/pt")
async def predict_pt(
    pt_src_image: UploadFile = File(..., description="Target pose person image"),
    pt_ref_image: UploadFile = File(..., description="Source person image"),
    pt_ref_acceleration: bool = Form(False, description="Accelerate Reference UNet"),
    pt_step: int = Form(30, description="Inference steps", ge=30, le=100),
    pt_scale: float = Form(2.5, description="Guidance scale", ge=0.1, le=5.0),
    pt_seed: int = Form(42, description="Random seed", ge=-1, le=2147483647),
    authenticated: bool = Depends(verify_authentication)
):
    """
    Secured endpoint for pose transfer prediction.
    Compatible with Gradio interface parameters.
    """
    
    # Use context manager for automatic cleanup
    with temp_files(pt_src_image, pt_ref_image) as (src_image_path, ref_image_path):
        try:
            generated_image, mask, densepose = leffa_predictor.leffa_predict_pt(
                src_image_path, ref_image_path, pt_ref_acceleration,
                pt_step, pt_scale, pt_seed
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error during model inference: " + str(e))

        response = {
            "generated_image": np_to_base64(generated_image),
            "mask": np_to_base64(mask),
            "densepose": np_to_base64(densepose),
        }
        return JSONResponse(content=response)

# Additional secured endpoints
@app.post("/predict/vt/simple")
async def predict_vt_simple(
    person_image: UploadFile = File(..., description="Person image"),
    garment_image: UploadFile = File(..., description="Garment image"),
    model_type: str = Form("viton_hd", description="Model type"),
    garment_type: str = Form("upper_body", description="Garment type"),
    steps: int = Form(30, description="Inference steps"),
    guidance_scale: float = Form(2.5, description="Guidance scale"),
    seed: int = Form(42, description="Random seed"),
    repaint: bool = Form(False, description="Repaint mode"),
    preprocess: bool = Form(False, description="Preprocess garment"),
    authenticated: bool = Depends(verify_authentication)
):
    """
    Secured simplified endpoint with more intuitive parameter names.
    """
    
    with temp_files(person_image, garment_image) as (person_path, garment_path):
        try:
            generated_image, mask, densepose = leffa_predictor.leffa_predict_vt(
                person_path, garment_path, False,  # ref_acceleration
                steps, guidance_scale, seed, model_type,
                garment_type, repaint, preprocess
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error during model inference: " + str(e))

        response = {
            "generated_image": np_to_base64(generated_image),
            "mask": np_to_base64(mask),
            "densepose": np_to_base64(densepose),
        }
        return JSONResponse(content=response)

@app.post("/predict/pt/simple")
async def predict_pt_simple(
    source_person: UploadFile = File(..., description="Source person image"),
    target_pose: UploadFile = File(..., description="Target pose person image"),
    steps: int = Form(30, description="Inference steps"),
    guidance_scale: float = Form(2.5, description="Guidance scale"),
    seed: int = Form(42, description="Random seed"),
    authenticated: bool = Depends(verify_authentication)
):
    """
    Secured simplified endpoint for pose transfer with intuitive parameter names.
    """
    
    with temp_files(target_pose, source_person) as (target_path, source_path):
        try:
            generated_image, mask, densepose = leffa_predictor.leffa_predict_pt(
                target_path, source_path, False,  # ref_acceleration
                steps, guidance_scale, seed
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error during model inference: " + str(e))

        response = {
            "generated_image": np_to_base64(generated_image),
            "mask": np_to_base64(mask),
            "densepose": np_to_base64(densepose),
        }
        return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=False)