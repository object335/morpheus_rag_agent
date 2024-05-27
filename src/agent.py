from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from transformers import LlamaTokenizer
from config import Config
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    set_global_tokenizer,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.indices.base import BaseIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.chat_engine.types import ChatMode
from llama_index.agent import ReActAgent
from llama_index.chat_engine import *
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

prompt="""Provided context:

{context_str}

You will be answering questions only based on the provided context.
If the answer is not provided in the context say i do not know the answer.
You will be answering user question below."""

def set_tokenizer():
    llama_tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=Config.model_path_hf,
        token=Config.api_token_hf,
    )
    set_global_tokenizer(llama_tokenizer.encode)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_tokenizer()

def init_service_context(model_location) -> ServiceContext:
    # Can be another model, used LlamaCPP since it's designed to run 
    # the LLaMA model using 4-bit integer quantization on a MacBook
    llm = LlamaCPP(
        model_path=model_location,
        temperature=0,
        max_new_tokens=512,
        # llama2 has a context window of 4096 tokens, recommended to set a little lower
        context_window=3900,
        # kwargs to pass to __call__()
         generate_kwargs = {"stop": ["<s>", "[INST]", "[/INST]"]},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 0},
        # transform inputs into Llama2 format with <INST> and <SYS> tags
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True
    )

    embed_model = HuggingFaceEmbedding(
        model_name=Config.embedding_model_name
    )
    return ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

def create_index(documents, service_context: ServiceContext) -> BaseIndex:
    return VectorStoreIndex.from_documents(documents, service_context=service_context)


# llm context initialization
service_context = init_service_context(Config.model_location)

def get_response(message):
    global agent
    chat_response = agent.chat(message.strip())
    return chat_response.response


data = "" 
agent = None
messages = []

@app.route('/upload', methods=['POST'])
def upload_file():
    global data,agent,service_context
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_type = file.filename.rsplit('.', 1)[1].lower()
    if file:
        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            documents = SimpleDirectoryReader(app.config['UPLOAD_FOLDER'], recursive=True).load_data()
            index = create_index(documents, service_context)
            agent = index.as_chat_engine(
                service_context=service_context, 
                context_prompt=prompt,
                skip_condense=True,
                chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, 
                verbose=True
            )
            return jsonify({"response":"You have successfully uploaded the text"})
        except Exception as e:
            print(str(e))
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File processing failed'}), 500

@app.route('/', methods=['POST'])
def chat():
    global messages
    try:
        data = request.get_json()
        if 'prompt' in data:
            prompt = data['prompt']['content']
            messages.append(data['prompt'])
            role = "assistant"
            if agent:
                response = get_response(prompt)
            else:
                response = "please upload a file first"
            messages.append({"role":role,"content":response})
            return jsonify({"role":role,"content":response})
        else:
            return jsonify({"error": "Missing required parameters"}), 400

    except Exception as e:
        return jsonify({"Error": str(e)}), 500 

@app.route('/messages', methods=['GET'])    
def get_messages():
    global messages
    try:
        return jsonify({"messages": messages})
    except Exception as e:
        return jsonify({"Error": str(e)}), 500 

@app.route('/clear_messages', methods=['GET'])    
def clear_messages():
    global messages
    try:
#        messages = [{'role':"assistant","content":"This highly experimental chatbot is not intended for making important decisions, and its responses are generated based on incomplete data and algorithms that may evolve rapidly. By using this chatbot, you acknowledge that you use it at your own discretion and assume all risks associated with its limitations and potential errors."}]
        messages = []
        return jsonify({"response": "successfully cleared message history"})
    except Exception as e:
        return jsonify({"Error": str(e)}), 500 
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)