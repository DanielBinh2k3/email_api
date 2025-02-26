from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()  # Load environment variables

# --- Google Generative AI Setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set")
genai.configure(api_key=GOOGLE_API_KEY)
#Using gemini-2.0-flash for this example.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

# --- Pydantic Models ---
class ContactInfo(BaseModel):
    phone: str
    email: EmailStr

class SalesInfo(BaseModel):
    name: str
    title: str
    contact_info: ContactInfo

class CustomerInfo(BaseModel):
    name: str
    title: str
    company: str
    contact_info: ContactInfo

class EmailParams(BaseModel):
    sales_info: SalesInfo
    customer_info: CustomerInfo
    emailContext: str
    tone: str
    length: str

class EmailRefinementRequest(BaseModel):  # Separate model for refinement
    emailContent: str
    refinementType: str
    suggestions: Optional[str] = None  # Make suggestions optional with default None

class EmailScoreRequest(BaseModel):
    emailContent: str

class EmailEvaluation(BaseModel):
    scores: Dict[str, int]
    suggestions: str

# --- FastAPI Setup ---
app = FastAPI()

# --- CORS Configuration (Important for Frontend Communication) ---
origins = [
    "http://localhost:3000",  # Allow your React frontend (adjust if needed)
    "http://localhost:3001", # Add other allowed origins here, if any
    "http://localhost:8000",
    "https://email-automation-mbw.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- API Endpoints ---
@app.post("/api/generate-email")
async def generate_email(email_params: EmailParams = Body(...)):
    try:
        # Using JSON output parser
        json_parser = JsonOutputParser()
        
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

        Vui lòng trả về JSON với định dạng như sau:
        {{
          "generatedEmail": "Nội dung email hoàn chỉnh ở đây"
        }}
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | json_parser
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

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refine-email")
async def refine_email(refinement_request: EmailRefinementRequest = Body(...)):
    try:
        refinement_type = refinement_request.refinementType
        email_content = refinement_request.emailContent
        
        # Using JSON output parser
        json_parser = JsonOutputParser()

        if refinement_type == 'professional':
            prompt_template = """
            Chỉnh sửa email sau để làm cho nó chuyên nghiệp hơn:
            
            {email_content}
            
            Trả về kết quả dưới dạng JSON với định dạng:
            {{
              "refinedEmail": "Nội dung email đã được chỉnh sửa ở đây"
            }}
            """
        elif refinement_type == 'shorter':
            prompt_template = """
            Chỉnh sửa email sau để làm cho nó ngắn gọn và súc tích hơn:
            
            {email_content}
            
            Trả về kết quả dưới dạng JSON với định dạng:
            {{
              "refinedEmail": "Nội dung email đã được chỉnh sửa ở đây"
            }}
            """
        elif refinement_type == 'personalized':
            prompt_template = """
            Chỉnh sửa email sau để làm cho nó cá nhân hóa hơn, xưng hô với người nhận bằng tên và đề cập đến công ty của họ:
            
            {email_content}
            
            Trả về kết quả dưới dạng JSON với định dạng:
            {{
              "refinedEmail": "Nội dung email đã được chỉnh sửa ở đây"
            }}
            """
        elif refinement_type == 'improvement':  #For auto suggestion
            prompt_template = """
            Dựa trên phản hồi của chuyên gia, hãy viết lại và cải thiện email sau. Cố gắng đạt được điểm số hoàn hảo.

            Email gốc:
            {email_content}

            Phản hồi và đề xuất ngắn gọn:
            {suggestions}

            Trả về kết quả dưới dạng JSON với định dạng:
            {{
              "refinedEmail": "Nội dung email đã cải thiện ở đây"
            }}
            """
        else:
            raise HTTPException(status_code=400, detail="Invalid refinement type")

        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | json_parser
        
        if refinement_type == 'improvement': # Pass suggestions to prompt
            if not refinement_request.suggestions:
                raise HTTPException(status_code=400, detail="Suggestions required for improvement refinement type")
            result = chain.invoke({"email_content": email_content, "suggestions": refinement_request.suggestions})
        else:
            result = chain.invoke({"email_content": email_content})

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/score-email")
async def score_email_endpoint(request: EmailScoreRequest):
    try:
        # Using JSON output parser
        json_parser = JsonOutputParser()
        
        prompt_template = """
        Bạn là một chuyên gia đánh giá email. Hãy đánh giá email sau và cung cấp:
        1. Điểm số từ 0 đến 10 cho mỗi tiêu chí sau:
           - Tiêu đề (subjectLine)
           - Cách viết (writingStyle)
           - Nội dung (content)
           - Cấu trúc và định dạng (structure)
           - Cá nhân hóa (personalization)
        2. Đề xuất cải thiện *ngắn gọn* cho email. Tóm tắt các ý chính, không cần giải thích dài dòng.
        
        Trả về kết quả dưới dạng JSON với định dạng:
        {{
          "scores": {{
            "subjectLine": [Điểm số từ 0-10],
            "writingStyle": [Điểm số từ 0-10],
            "content": [Điểm số từ 0-10], 
            "structure": [Điểm số từ 0-10],
            "personalization": [Điểm số từ 0-10]
          }},
          "suggestions": "Đề xuất cải thiện ngắn gọn ở đây"
        }}

        Email cần đánh giá:
        {email_content}
        """

        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the chain
        chain = prompt | llm | json_parser
        
        # Invoke the chain with the email content
        evaluation_result = chain.invoke({"email_content": request.emailContent})
        
        return evaluation_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring email: {str(e)}")