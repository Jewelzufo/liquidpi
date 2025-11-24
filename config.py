"""Configuration management for the Dual-Agent System"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Centralized configuration management"""
    
    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    # Models
    generator_model: str = os.getenv(
        "GENERATOR_MODEL", 
        "hf.co/unsloth/granite-4.0-350m-GGUF:Q4_K_M"
    )
    critic_model: str = os.getenv(
        "CRITIC_MODEL", 
        "hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M"
    )
    
    # Behavior
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
    enable_thinking_display: bool = os.getenv("ENABLE_THINKING_DISPLAY", "true").lower() == "true"
    enable_rich_ui: bool = os.getenv("ENABLE_RICH_UI", "true").lower() == "true"
    
    # Generation parameters
    temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.5"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE")

# Global config instance
config = Config()
