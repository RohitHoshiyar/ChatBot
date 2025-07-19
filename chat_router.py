from fastapi import APIRouter
from pydantic import BaseModel
from recommender import recommend_for_user, semantic_search
from gpt_utils import generate_gpt_response
from utils import detect_intent, get_order_status, get_return_policy

router = APIRouter()

class ChatInput(BaseModel):
    user_id: str
    message: str

@router.post("/chat")
def chat(input: ChatInput):
    intent = detect_intent(input.message)

    if intent == "recommendation":
        return {"reply": recommend_for_user(input.user_id, input.message)}
    elif intent == "order_status":
        return {"reply": get_order_status(input.user_id)}
    elif intent == "return_policy":
        return {"reply": get_return_policy()}
    else:
        return {"reply": generate_gpt_response(input.message)}
