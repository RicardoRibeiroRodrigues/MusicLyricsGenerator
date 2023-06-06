from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from models_logic import generate_lyrics
from random import randint


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set the appropriate origin(s) for your application
    allow_methods=["POST"],  # Add any other HTTP methods your API supports
    allow_headers=["*"],  # You can restrict the headers if needed
)

class TextRequest(BaseModel):
    text: str
    n: int = 20
    model_n: int = 0

class TextResponse(BaseModel):
    generated_text: str
    model_used: str


@app.get("/")
def read_root():
    return "Go to /docs to see the documentation!"


@app.get("/random_snippet")
def random_snippet() -> str:
    """
    Returns a random snippet from the test dataset
    """
    random_line = randint(0, 270_000)
    with open('snippets.csv', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == random_line:
                return line
    

@app.post("/generate_text")
def generate_text_endpoint(request: TextRequest) -> TextResponse:
    """
    Generates text using a random model
    """
    res = generate_lyrics(request.text, request.n, request.model_n)
    response = TextResponse(generated_text=res[0], model_used=res[1])
    print(response)
    return response
