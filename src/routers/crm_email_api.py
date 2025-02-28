from fastapi import APIRouter, HTTPException, Body
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.schemas import EmailParams, EmailRefinementRequest, EmailScoreRequest
from src.dependencies import get_llm
from src.internal.llm import build_format_instructions
from src.config import AVAILABLE_MODELS

router = APIRouter(prefix="/api", tags=["emails"])

@router.get("/models")
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

@router.post("/generate-email")
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

@router.post("/refine-email")
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

@router.post("/score-email")
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