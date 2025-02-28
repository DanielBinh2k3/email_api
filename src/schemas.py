from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field, EmailStr
from pydantic_extra_types.phone_numbers import PhoneNumber

class ContactInfo(BaseModel):
    phone: str
    email: EmailStr

class SalesInfo(BaseModel):
    name: str = Field(..., min_length=2)
    title: str
    contact_info: ContactInfo

class CustomerInfo(BaseModel):
    name: str = Field(..., min_length=2)
    title: str
    company: str
    contact_info: ContactInfo

class EmailParams(BaseModel):
    sales_info: SalesInfo
    customer_info: CustomerInfo
    emailContext: str = Field(..., min_length=10)
    tone: str
    length: Literal["short", "medium", "long"]
    outputFormat: Optional[Literal["plain", "html", "markdown"]] = "markdown"
    model: Optional[str] = "gemini-1.5-pro"  # Will be imported from config
    temperature: Optional[float] = 0.7

class EmailRefinementRequest(BaseModel):
    emailContent: str = Field(..., min_length=10)
    refinementType: Literal["professional", "shorter", "personalized", "improvement"]
    suggestions: Optional[str] = None
    outputFormat: Optional[Literal["plain", "html", "markdown"]] = "markdown"
    model: Optional[str] = "gemini-1.5-pro"  # Will be imported from config
    temperature: Optional[float] = 0.7

class EmailScoreRequest(BaseModel):
    emailContent: str = Field(..., min_length=10)
    outputFormat: Optional[Literal["plain", "html", "markdown"]] = "markdown"
    model: Optional[str] = "gemini-1.5-pro"  # Will be imported from config
    temperature: Optional[float] = 0.7

class ModelInfo(BaseModel):
    name: str
    provider: str
    description: str