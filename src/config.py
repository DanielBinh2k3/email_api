import os
from dotenv import load_dotenv

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