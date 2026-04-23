"""
Agri-Trust | FastAPI Server
=============================
Main API server for the Agri-Trust B2B crop verification platform.

Endpoints:
  POST /verify-crop   →  Full AI-powered crop inspection pipeline
  GET  /health        →  Health check
"""

import os
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ── ML Module Imports ─────────────────────────────────────────────────────────
from ml.quality_grader import CropQualityGrader
from ml.feature_extractor import CropFeatureExtractor
from ml.pricing_engine import PricingEngine


# ══════════════════════════════════════════════════════════════════════════════
#  App Initialization
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Agri-Trust API",
    description="AI-Driven B2B Trade Protocol for Bulk Agriculture — "
                "Crop verification, quality grading, and fair pricing.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize ML Models (loaded once at startup) ────────────────────────────
print("🚀  Initializing Agri-Trust AI Modules...")
quality_grader = CropQualityGrader()
feature_extractor = CropFeatureExtractor()
pricing_engine = PricingEngine()
print("✅  All AI modules initialized.\n")

# In-memory store for verified harvests (use a DB in production)
verified_harvests: List[dict] = []


# ══════════════════════════════════════════════════════════════════════════════
#  Health Check
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
async def health_check():
    """Returns server status and loaded model metadata."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": {
            "quality_grader": True,
            "feature_extractor": True,
            "pricing_engine": pricing_engine._is_trained,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  POST /verify-crop — Full AI Inspection Pipeline
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/verify-crop")
async def verify_crop(
    file: UploadFile = File(..., description="Crop image (JPEG/PNG)"),
    bulk_volume_tons: Optional[float] = Form(10.0, description="Shipment volume in tons"),
    regional_market_index: Optional[float] = Form(1.0, description="Regional demand index (0.5 – 2.0)"),
    seasonality_factor: Optional[float] = Form(1.0, description="Season adjustment (0.5 – 1.5)"),
    harvest_location: Optional[str] = Form("Unknown", description="GPS / region of harvest"),
    crop_type: Optional[str] = Form("General", description="Type of crop (e.g. Corn, Wheat)"),
    language: Optional[str] = Form("en-IN", description="Language code (e.g. en-IN, hi-IN, te-IN)"),
):
    """
    **Full AI-Powered Crop Verification**

    Accepts a crop image and optional market parameters, then runs:
    1. **Quality Grading (CNN)** → MobileNetV2 classification
    2. **Feature Extraction (CV)** → Defect quantification
    3. **Pricing Engine (ML)** → Fair market valuation
    Includes Quality Grading, Feature Extraction, Pricing, and **Disease Diagnosis**.

    Returns a complete verification report.
    """
    # ── Validate upload ───────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    # ── 1. Quality Grading ────────────────────────────────────────────────
    try:
        grading_result = quality_grader.grade(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quality grading failed: {str(e)}",
        )

    # ── 2. Feature Extraction ─────────────────────────────────────────────
    try:
        defect_analysis = feature_extractor.extract(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}",
        )

    # ── 3. Pricing ────────────────────────────────────────────────────────
    try:
        estimated_valuation = pricing_engine.predict_price(
            quality_score=grading_result["purity_score"],
            bulk_volume_tons=bulk_volume_tons,
            regional_market_index=regional_market_index,
            seasonality_factor=seasonality_factor,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pricing prediction failed: {str(e)}",
        )

    # ── 4. Disease Diagnosis (Simulated Knowledge Base) ───────────────────
    # In a real system, this would be another ML model classification.
    # Here uses Logic + Knowledge Base based on crop_type + quality.
    disease_info = None
    if grading_result["purity_score"] < 0.85:
        # If quality is low, assume disease/issue present
        disease_info = get_disease_diagnosis(crop_type, lang=language)

    # ── Build Response ────────────────────────────────────────────────────
    certificate_id = str(uuid.uuid4())[:12].upper()
    timestamp = datetime.now(timezone.utc).isoformat()

    response = {
        "certificate_id": f"AT-{certificate_id}",
        "quality_grade": grading_result["quality_grade"],
        "purity_score": grading_result["purity_score"],
        "estimated_valuation": estimated_valuation,
        "defect_analysis": defect_analysis,
        "disease_diagnosis": disease_info,
        "metadata": {
            "confidence_scores": grading_result["confidence_scores"],
            "bulk_volume_tons": bulk_volume_tons,
            "regional_market_index": regional_market_index,
            "seasonality_factor": seasonality_factor,
            "harvest_location": harvest_location,
            "crop_type": crop_type,
            "timestamp": timestamp,
            "feature_importance": pricing_engine.get_feature_importance(),
        },
    }

    # Store in memory for the buyer dashboard
    verified_harvests.append(response)

    return JSONResponse(content=response)


def get_disease_diagnosis(crop_type: str, lang: str = "en-IN") -> dict:
    """Mock Knowledge Base for Disease Diagnosis with Multilingual Support."""
    
    # English DB
    db_en = {
        "corn": {
            "name": "Northern Corn Leaf Blight",
            "symptoms": "Grayish-green elliptical lesions on leaves.",
            "cause": "Fungal pathogen Exserohilum turcicum.",
            "solution": "Apply fungicides like Mancozeb or Propiconazole. Rotation with non-host crops.",
            "prevention": "Use resistant hybrids and manage residue."
        },
        "wheat": {
            "name": "Yellow Rust (Stripe Rust)",
            "symptoms": "Yellow streaks (pustules) on leaves.",
            "cause": "Fungal pathogen Puccinia striiformis.",
            "solution": "Spray Tebuconazole or Propiconazole immediately.",
            "prevention": "Plant resistant varieties and remove volunteer wheat."
        },
        "tomato": {
            "name": "Early Blight",
            "symptoms": "Concentric rings (target pattern) on older leaves.",
            "cause": "Alternaria solani fungus.",
            "solution": "Apply Copper-based fungicides or Chlorothalonil.",
            "prevention": "Mulch soil to prevent splashes, ensure good airflow."
        },
        "soybean": {
            "name": "Soybean Rust",
            "symptoms": "Small brown pustules on underside of leaves.",
            "cause": "Phakopsora pachyrhizi fungus.",
            "solution": "Apply Pyraclostrobin or Azoxystrobin fungicides.",
            "prevention": "Early detection and scouting are critical."
        },
        "potato": {
            "name": "Late Blight",
            "symptoms": "Dark, water-soaked spots on leaves with white mold.",
            "cause": "Phytophthora infestans (Oomycete).",
            "solution": "Apply Metalaxyl or Mancozeb. Destroy infected tubers.",
            "prevention": "Use certified disease-free seed tubers."
        }
    }

    # Hindi DB
    db_hi = {
        "corn": {
            "name": "उत्तरी मक्का पत्ती झुलसा रोग (Northern Corn Leaf Blight)",
            "symptoms": "पत्तियों पर भूरे-हरे अंडाकार घाव।",
            "cause": "फफूंद रोगजनक (Exserohilum turcicum)।",
            "solution": "मैंकोज़ेब या प्रोपिकोनाज़ोल जैसे कवकनाशी का प्रयोग करें। फसल चक्र अपनाएं।",
            "prevention": "प्रतिरोधी किस्में उपयोग करें और अवशेष प्रबंधन करें।"
        },
        "wheat": {
            "name": "पीला रतुआ (Yellow Rust)",
            "symptoms": "पत्तियों पर पीली धारियां।",
            "cause": "फफूंद (Puccinia striiformis)।",
            "solution": "टेबुकोनाज़ोल या प्रोपिकोनाज़ोल का तुरंत छिड़काव करें।",
            "prevention": "प्रतिरोधी किस्में बोएं।"
        },
        "tomato": {
            "name": "अगेती झुलसा (Early Blight)",
            "symptoms": "पुरानी पत्तियों पर गोल छल्लेदार धब्बे।",
            "cause": "अल्टरनेरिया सोलाई कवक।",
            "solution": "कॉपर-आधारित कवकनाशी या क्लोरोथलोनिल का प्रयोग करें।",
            "prevention": "मल्चिंग करें और हवा का प्रवाह सुनिश्चित करें।"
        }
    }

    # Telugu DB
    db_te = {
        "corn": {
            "name": "మొక్కజొన్న ఆకు మచ్చ తెగులు (Leaf Blight)",
            "symptoms": "ఆకులపై బూడిద-ఆకుపచ్చ రంగు మచ్చలు.",
            "cause": "శిలీంధ్ర వ్యాధి.",
            "solution": "మాంకోజెబ్ లేదా ప్రొపికొనజోల్ పిచికారీ చేయండి.",
            "prevention": "తెగులు తట్టుకునే రకాలను వాడండి."
        },
        "wheat": {
            "name": "గోధుమ కుంకుమ తెగులు (Yellow Rust)",
            "symptoms": "ఆకులపై పసుపు చారలు.",
            "cause": "శిలీంధ్రం.",
            "solution": "టెబుకొనజోల్ వెంటనే పిచికారీ చేయండి.",
            "prevention": "నిరోధక రకాలను ఎంచుకోండి."
        },
        "tomato": {
            "name": "టొమాటో మాడు తెగులు (Early Blight)",
            "symptoms": "ముదురు ఆకులపై వలయాకారపు మచ్చలు.",
            "cause": "శిలీంధ్రం.",
            "solution": "కాపర్ ఆక్సీక్లోరైడ్ పిచికారీ చేయండి.",
            "prevention": "పొలంలో గాలి ప్రసరణ బాగుండాలి."
        }
    }

    # Normalize key
    key = crop_type.lower() if crop_type else "general"

    # Select DB based on lang (start check)
    selected_db = db_en # Default
    if lang.startswith("hi"):
        selected_db = db_hi
    elif lang.startswith("te"):
        selected_db = db_te

    # Fallback if crop not in localized DB, try English
    data = selected_db.get(key)
    if not data and selected_db != db_en:
        data = db_en.get(key)
        
    if not data:
        # Generic Fallback
        if lang.startswith("hi"):
             return {
                "name": "सामान्य तनाव / पोषक तत्वों की कमी",
                "symptoms": "पत्तियों का रंग बदलना या मुरझाना।",
                "cause": "पानी की कमी, नाइट्रोजन की कमी या मामूली कीट।",
                "solution": "मिट्टी की नमी और NPK स्तर की जाँच करें। संतुलित उर्वरक डालें।",
                "prevention": "नियमित मिट्टी परीक्षण और सिंचाई प्रबंधन।"
            }
        elif lang.startswith("te"):
             return {
                "name": "సాధారణ పోషక లోపం",
                "symptoms": "ఆకులు రంగు మారడం లేదా వాడిపోవడం.",
                "cause": "నీటి ఎద్దడి లేదా నైట్రోజన్ లోపం.",
                "solution": "నేల తేమ మరియు ఎరువులను పరీక్షించండి.",
                "prevention": "క్రమం తప్పకుండా నీటి పారుదలని గమనించండి."
            }
        else:
             return {
                "name": "General Stress / Nutrient Deficiency",
                "symptoms": "Discoloration or wilting of leaves.",
                "cause": "Could be water stress, nitrogen deficiency, or minor pest damage.",
                "solution": "Check soil moisture and NPK levels. Apply balanced fertilizer.",
                "prevention": "Regular soil testing and irrigation management."
            }

    return data


# ══════════════════════════════════════════════════════════════════════════════
#  GET /verified-harvests — Buyer Dashboard Data
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/verified-harvests")
async def get_verified_harvests():
    """
    Returns all verified harvests sorted by AI-confirmed quality score
    (highest first) for the Buyer Dashboard.
    """
    sorted_harvests = sorted(
        verified_harvests,
        key=lambda h: h["purity_score"],
        reverse=True,
    )
    return {"total": len(sorted_harvests), "harvests": sorted_harvests}


# ══════════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
