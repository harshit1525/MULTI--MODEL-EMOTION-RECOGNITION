from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    status: str = Field("success", description="Status of prediction: success or error")
    modality: str = Field(..., description="Modality analyzed: audio, text, video, or multimodal")
    predicted_class: str = Field(..., description="Predicted emotion label")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class-wise probability distribution")
    model_used: str = Field(..., description="Identifier of the model used")
    inference_time_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional modality-specific metadata")

class MultimodalPredictionResponse(BaseModel):
    status: str = Field("success", description="Status of multimodal prediction")
    fused_predicted_class: Optional[str] = Field(None, description="ONE prominent unified final predicted emotion")
    fused_confidence: Optional[float] = Field(None, description="Overall calibrated unified confidence score")
    fused_probabilities: Optional[Dict[str, float]] = Field(None, description="Combined probability distribution across all 7 discrete emotions")
    fusion_method: str = Field(..., description="Name/Architecture of the fusion engine")
    active_modalities: List[str] = Field(default_factory=list, description="List of modalities that contributed to inference")
    modality_weights: Dict[str, float] = Field(default_factory=dict, description="Dynamic attention / reliability weights per modality")
    modality_coverage: Dict[str, bool] = Field(default_factory=dict, description="Coverage indicators for audio, text, video")
    modality_agreement: str = Field(..., description="Full Consensus, Partial Consensus, or Conflicting / Divergent")
    agreement_analysis: str = Field(..., description="Summary of agreement or tension between modalities")
    unified_explanation: str = Field(..., description="In-depth scientific affect rationale and conflict resolution narrative")
    conflict_analysis: Optional[str] = Field(None, description="Specific breakdown of any modality disagreements")
    transcript_used: Optional[str] = Field("", description="Spoken or written transcript that was analyzed")
    temperature_scaling: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Temperature scaling and probability calibration metadata")
    quality_indicators: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Signal quality metrics and reliability assessments for each modality")
    individual_predictions: Dict[str, Optional[Any]] = Field(default_factory=dict, description="Secondary diagnostic outputs from individual models")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or missing modality notices")

class HealthResponse(BaseModel):
    status: str = "healthy"
    device: str
    models_loaded: Dict[str, bool]
    supported_emotions: List[str]
    version: str = "2.0.0"
