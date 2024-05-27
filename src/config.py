import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Configuration object
class Config:
    PATH = ""
    model_location="model/llama-2-7b-chat.Q4_K_M.gguf"
    model_name="TheBloke/Llama-2-7B-Chat-GGUF"
    model_revision="llama-2-7b-chat.Q4_K_M.gguf"
    embedding_model_name="BAAI/bge-small-en-v1.5"
    model_path_hf="meta-llama/Llama-2-7b-chat-hf"
    api_token_hf="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"