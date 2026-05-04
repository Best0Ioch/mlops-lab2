from pydantic import BaseModel, Field

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="cm")
    sepal_width: float = Field(..., ge=0, le=10, description="cm")
    petal_length: float = Field(..., ge=0, le=10, description="cm")
    petal_width: float = Field(..., ge=0, le=10, description="cm")

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float