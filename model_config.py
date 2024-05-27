import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Configuration object
class Config:
    # Model configuration
    MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGUF"
    MODEL_REVISION = "llama-2-7b-chat.Q4_K_M.gguf"
    MODEL_PATH = "model/"+MODEL_REVISION
    DOWNLOAD_DIR = "model"
        