def detect_intent(text: str) -> str:
    text = text.lower()
    if "recommend" in text or "suggest" in text:
        return "recommendation"
    elif "track" in text or "order" in text:
        return "order_status"
    elif "return" in text or "refund" in text:
        return "return_policy"
    else:
        return "general"

def get_order_status(user_id: str) -> str:
    return f"Order for user {user_id} is in transit. It will arrive within 3-5 days."

def get_return_policy() -> str:
    return "You can return any item within 15 days of delivery. Visit /returns to start."
