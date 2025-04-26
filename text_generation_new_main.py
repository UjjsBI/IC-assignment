# filename: app.py

from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch # type: ignore

app = FastAPI(title="Mistral-7B Text Generation API")

# Load Mistral-7B-Instruct model and tokenizer
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
    raise RuntimeError(f"Failed to load Mistral-7B model: {str(e)}")

# Request body schema
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

# Route to generate text
@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        result = generator(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.95
        )
        return {"response": result[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
