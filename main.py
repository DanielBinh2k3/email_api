from typing import Optional, List, Dict, Literal
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, constr
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
# from pydantic_extra_types.phone_numbers import PhoneNumber
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM, BaseLLM


# Custom class for OpenRouter
class ChatOpenRouter(ChatOpenAI):
    """Custom class for OpenRouter integration."""
    
    def __init__(
        self, 
        model_name: str, 
        openai_api_key: Optional[str] = None, 
        openai_api_base: str = "https://openrouter.ai/api/v1", 
        **kwargs
    ):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        super().__init__(
            model=model_name,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            **kwargs
        )

load_dotenv()  # Load environment variables

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Define available models
AVAILABLE_MODELS = {
    # Google Gemini models
    "gemini-1.5-pro": {
        "provider": "google",
        "description": "Google's Gemini 1.5 Pro model"
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "description": "Google's Gemini 1.5 Flash model"
    },
    # OpenRouter models
    "google/gemini-2.0-flash-thinking-exp:free": {
        "provider": "openrouter",
        "description": "Google Gemini 2.0 Flash Thinking (Experimental)"
    },
    "google/gemini-2.0-pro-exp-02-05:free": {
        "provider": "openrouter",
        "description": "Google Gemini 2.0 Pro (Experimental)"
    },
    "google/gemini-2.0-flash-exp:free": {
        "provider": "openrouter",
        "description": "Google Gemini 2.0 Flash (Experimental)"
    },
    "google/learnlm-1.5-pro-experimental:free": {
        "provider": "openrouter",
        "description": "Google LearnLM 1.5 Pro (Experimental)"
    },
    "deepseek/deepseek-r1-distill-llama-70b:free": {
        "provider": "openrouter",
        "description": "DeepSeek R1 Distill LLaMA 70B"
    },
    "deepseek/deepseek-chat:free": {
        "provider": "openrouter",
        "description": "DeepSeek Chat"
    },
    "deepseek/deepseek-r1:free": {
        "provider": "openrouter",
        "description": "DeepSeek R1"
    }
}

# Default model
DEFAULT_MODEL = "gemini-1.5-pro"

# --- Model Factory Function ---
def get_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.7, **kwargs):
    """Create an LLM instance based on the specified model name."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Use one of: {', '.join(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_name]
    
    if model_info["provider"] == "google":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            **kwargs
        )
    
    elif model_info["provider"] == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown provider: {model_info['provider']}")

# --- Pydantic Models ---
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
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.7

class EmailRefinementRequest(BaseModel):
    emailContent: str = Field(..., min_length=10)
    refinementType: Literal["professional", "shorter", "personalized", "improvement"]
    suggestions: Optional[str] = None
    outputFormat: Optional[Literal["plain", "html", "markdown"]] = "markdown"
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.7

class EmailScoreRequest(BaseModel):
    emailContent: str = Field(..., min_length=10)
    outputFormat: Optional[Literal["plain", "html", "markdown"]] = "markdown"
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.7

class ModelInfo(BaseModel):
    name: str
    provider: str
    description: str

# --- FastAPI Setup ---
app = FastAPI(title="Email Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def build_format_instructions(output_format: str) -> str:
    """Generate format-specific instructions for prompts"""
    instructions = {
        "html": (
            "Vui lòng tạo email với định dạng HTML. Bao gồm các thẻ HTML cơ bản như <p>, <br>, <strong>, <em>, v.v. "
            "Đảm bảo các đoạn văn được bọc trong thẻ <p> và ngắt dòng sử dụng <br>.\n\n"
            "Trả về kết quả dưới định dạng JSON với trường 'generatedEmail' chứa nội dung HTML."
        ),
        "markdown": (
            "Vui lòng tạo email với định dạng Markdown. Sử dụng cú pháp markdown cơ bản như **, *, #, >, etc.\n\n"
            "Trả về kết quả dưới định dạng JSON với trường 'generatedEmail' chứa nội dung Markdown."
        ),
        "plain": (
            "Vui lòng tạo email với định dạng văn bản thuần túy, không có bất kỳ thẻ đánh dấu nào.\n\n"
            "Trả về kết quả dưới định dạng JSON với trường 'generatedEmail' chứa văn bản thuần túy."
        )
    }
    return instructions.get(output_format, instructions["plain"])

# --- API Endpoints ---
@app.get("/api/models")
async def list_models():
    """List all available LLM models."""
    result = []
    for name, info in AVAILABLE_MODELS.items():
        result.append({
            "name": name,
            "provider": info["provider"],
            "description": info["description"]
        })
    return {"models": result}

@app.post("/api/generate-email")
async def generate_email(email_params: EmailParams = Body(...)):
    try:
        # Get the requested LLM
        llm = get_llm(email_params.model, email_params.temperature)
        
        prompt_template = """
        Tạo một email dựa trên các thông tin sau:
        
        Thông tin người bán:
        - Tên: {sales_name}
        - Chức vụ: {sales_title}
        - Số điện thoại: {sales_phone}
        - Email: {sales_email}
        
        Thông tin khách hàng:
        - Tên: {customer_name}
        - Chức vụ: {customer_title}
        - Công ty: {customer_company}
        - Số điện thoại: {customer_phone}
        - Email: {customer_email}
        
        Nội dung email:
        {email_context}
        
        Giọng văn mong muốn: {tone}
        Độ dài mong muốn: {length}
        
        {format_instructions}
        """
        
        format_instructions = build_format_instructions(email_params.outputFormat)
        prompt = PromptTemplate.from_template(prompt_template)
        
        chain = (
            RunnablePassthrough.assign(format_instructions=lambda _: format_instructions)
            | prompt
            | llm
            | JsonOutputParser()
        )
        
        result = chain.invoke({
            "sales_name": email_params.sales_info.name,
            "sales_title": email_params.sales_info.title,
            "sales_phone": email_params.sales_info.contact_info.phone,
            "sales_email": email_params.sales_info.contact_info.email,
            "customer_name": email_params.customer_info.name,
            "customer_title": email_params.customer_info.title,
            "customer_company": email_params.customer_info.company,
            "customer_phone": email_params.customer_info.contact_info.phone,
            "customer_email": email_params.customer_info.contact_info.email,
            "email_context": email_params.emailContext,
            "tone": email_params.tone,
            "length": email_params.length,
        })
        
        # Add model information to response
        result["model_used"] = email_params.model
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo email: {str(e)}")

@app.post("/api/refine-email")
async def refine_email(refinement_request: EmailRefinementRequest = Body(...)):
    try:
        # Get the requested LLM
        llm = get_llm(refinement_request.model, refinement_request.temperature)
        
        if refinement_request.refinementType == "improvement" and not refinement_request.suggestions:
            raise HTTPException(status_code=400, detail="Yêu cầu đề xuất cải thiện cần có thông tin gợi ý")
        
        base_prompts = {
            "professional": "Chỉnh sửa email sau để làm cho nó chuyên nghiệp hơn:\n\n{email_content}",
            "shorter": "Chỉnh sửa email sau để làm cho nó ngắn gọn và súc tích hơn:\n\n{email_content}",
            "personalized": "Chỉnh sửa email sau để làm cho nó cá nhân hóa hơn:\n\n{email_content}",
            "improvement": "Dựa trên phản hồi sau, hãy cải thiện email:\n\nGợi ý: {suggestions}\n\nEmail gốc:\n{email_content}"
        }
        
        prompt_template = base_prompts[refinement_request.refinementType]
        prompt_template += "\n\n" + build_format_instructions(refinement_request.outputFormat)
        
        chain = (
            PromptTemplate.from_template(prompt_template)
            | llm
            | JsonOutputParser()
        )
        
        inputs = {"email_content": refinement_request.emailContent}
        if refinement_request.refinementType == "improvement":
            inputs["suggestions"] = refinement_request.suggestions
        
        result = chain.invoke(inputs)
        
        # Add model information to response
        result["model_used"] = refinement_request.model
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=400, detail="Loại chỉnh sửa không hợp lệ")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi chỉnh sửa email: {str(e)}")

@app.post("/api/score-email")
async def score_email_endpoint(request: EmailScoreRequest):
    try:
        # Get the requested LLM
        llm = get_llm(request.model, request.temperature)
        
        prompt_template = """
        Đánh giá email theo các tiêu chí sau (0-10):
        - Tiêu đề: {subjectLine}
        - Phong cách viết: {writingStyle}
        - Nội dung: {content}
        - Cấu trúc: {structure}
        - Cá nhân hóa: {personalization}
        
        Đề xuất cải thiện: {suggestions}
        
        Email đánh giá:
        {email_content}
        
        {format_instructions}
        """
        
        format_instructions = build_format_instructions(request.outputFormat)
        
        chain = (
            RunnablePassthrough.assign(
                format_instructions=lambda _: format_instructions,
                subjectLine="Đánh giá tiêu đề email",
                writingStyle="Đánh giá phong cách viết", 
                content="Đánh giá nội dung chính",
                structure="Đánh giá cấu trúc email",
                personalization="Đánh giá mức độ cá nhân hóa",
                suggestions="Đề xuất cải thiện chi tiết"
            )
            | PromptTemplate.from_template(prompt_template)
            | llm
            | JsonOutputParser()
        )
        
        result = chain.invoke({"email_content": request.emailContent})
        
        # Add model information to response
        result["model_used"] = request.model
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đánh giá email: {str(e)}")
