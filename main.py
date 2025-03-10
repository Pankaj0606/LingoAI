from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ CORS Middleware (Allow both localhost and 127.0.0.1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Home route
@app.get("/", status_code=200)
def home():
    return {"message": "Translator API is running!"}

# ✅ Define request body format
class TranslationRequest(BaseModel):
    text: str
    target_lang: Literal["en", "fr", "es", "de"]  # Allowed languages

# ✅ Function to get translation model
def get_translation_model(source_lang: str, target_lang: str):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception:
        return None, None  # Graceful error handling

# ✅ Translation API
@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    detected_lang = detect(request.text)

    # ✅ Ensure source and target are not the same
    if detected_lang == request.target_lang:
        return {"error": "Source and target languages cannot be the same."}

    # ✅ Load correct model
    tokenizer, model = get_translation_model(detected_lang, request.target_lang)
    if not model:
        return {"error": f"Translation model not available for {detected_lang} to {request.target_lang}"}

    # ✅ Perform translation
    inputs = tokenizer(request.text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return {
        "original_language": detected_lang,
        "translated_text": translated_text
    }
