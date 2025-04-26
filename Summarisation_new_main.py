# filename: summarizer_app.py

from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch # type: ignore

app = FastAPI(title="Mistral-7B Summarization API")

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
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Request body schema
class SummarizationRequest(BaseModel):
    text: str
    max_tokens: int = 150
    temperature: float = 0.5

@app.post("/summarize")
def summarize_text(request: SummarizationRequest):
    try:
        prompt = f"Summarize the following text:\n\n{request.text}\n\nSummary:"
        result = generator(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.9
        )
        return {"summary": result[0]["generated_text"].replace(prompt, "").strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
