from pydantic import BaseModel

class ChurnInput(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    senior_citizen: int  # 0 or 1