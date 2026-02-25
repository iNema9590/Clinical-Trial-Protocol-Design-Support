import torch
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME ="microsoft/Phi-3.5-mini-instruct"
MODEL_NAME = "gemini-2.0-flash"  # Use "gemini-2.0-flash" or "gemini-1.5-pro" for Gemini models

# Check if using Gemini model
IS_GEMINI = MODEL_NAME.startswith("gemini-")

if IS_GEMINI:
       
    # Initialize Vertex AI
    vertexai.init(
        project="oag-ai",
        credentials=service_account.Credentials.from_service_account_file(
            "../google-credentials.json"
        ),
    )
    
    # Create Gemini model
    model = GenerativeModel(model_name=MODEL_NAME)
    tokenizer = None
else:
    # Load transformer model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

def generate(prompt, max_new_tokens=4096, temperature=0.1):
    if IS_GEMINI:
        # Generate using Gemini
        response = model.generate_content(
            contents=[prompt],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_new_tokens,
            }
        )
        return response.text
    else:
        # Generate using transformer model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)