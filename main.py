from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llms import rag_with_user_preferences

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class UserMessage(BaseModel):
    message: str
    user_id: str

@app.post("/chat")
async def chat(user_message: UserMessage):
    try:
        response_text = rag_with_user_preferences(user_message.user_id, user_message.message)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
