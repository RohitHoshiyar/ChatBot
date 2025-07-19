import streamlit as st
import requests

st.title("🛍️ AI E-commerce Chatbot")
user_id = st.text_input("Enter your User ID:")
user_msg = st.text_input("Type your query:")

if st.button("Send"):
    res = requests.post("http://localhost:8000/chat", json={
        "user_id": user_id,
        "message": user_msg
    })
    st.write("🤖:", res.json()['reply'])
