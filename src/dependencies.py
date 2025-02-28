from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, GOOGLE_API_KEY, OPENROUTER_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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