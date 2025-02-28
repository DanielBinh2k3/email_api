from typing import Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM, BaseLLM
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, GOOGLE_API_KEY, OPENROUTER_API_KEY

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
