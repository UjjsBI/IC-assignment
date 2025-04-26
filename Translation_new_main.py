# filename: translation_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI(title="Mistral Translation API")

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Request schema
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    max_tokens: int = 150
    temperature: float = 0.3

@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        # Example prompt: Translate from English to French:
        prompt = (
            f"Translate this from {request.source_lang} to {request.target_lang}:\n\n"
            f"{request.text}\n\nTranslation:"
        )
        result = generator(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.9,
            do_sample=True
        )
        translated = result[0]["generated_text"].replace(prompt, "").strip()
        return {"translation": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
