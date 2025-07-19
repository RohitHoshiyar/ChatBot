
# 🛍️ AI-Powered E-commerce Chatbot

A full-stack AI chatbot system for intelligent, personalized e-commerce support — combining vector search, cold-start logic, product recommendation, order tracking, return handling, and LLM-powered natural language responses. Perfect for API-based online shopping websites.

---

## 🌟 Features

- 🔎 **Smart Product Recommendations**
  - FAISS-based semantic search
  - Cold start logic using user demographics
  - Cosine similarity with collaborative filtering
- 💬 **Conversational AI with GPT**
  - Natural dialogue using OpenAI GPT or Azure GPT
  - Handles vague or general questions intelligently
- 📦 **Customer Support**
  - Order tracking by user ID
  - Return/refund policy Q&A
- 🧠 **Intent Detection**
  - Routes queries to the right module (Rec, Order, Return, GPT)
- 🧰 **Modular FastAPI Backend**
  - Easily pluggable into existing e-commerce websites
- 🖥️ **Optional Streamlit Chat UI**
  - For quick prototyping and testing

---

## 🧱 Project Structure

```
ai-ecommerce-chatbot/
├── backend/
│   ├── main.py               # FastAPI app
│   ├── chat_router.py        # Handles POST /chat
│   ├── recommender.py        # Vector search + cold start recs
│   ├── gpt_utils.py          # GPT-3.5/Azure GPT logic
│   ├── utils.py              # Intent detection, order/returns logic
│
├── data/
│   ├── products_large.csv
│   ├── ratings_large.csv
│   └── Updated_Users_Dataset_with_Demographics.csv
│
├── vector_index/
│   └── faiss_index.bin
│
├── frontend/
│   └── streamlit_chat.py     # Optional chat UI
│
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/ai-ecommerce-chatbot.git
cd ai-ecommerce-chatbot
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Ensure your 3 CSV files are in the `data/` folder:

- `products_large.csv`
- `ratings_large.csv`
- `Updated_Users_Dataset_with_Demographics.csv`

Also make sure your `faiss_index.bin` is in `vector_index/`.

### 4. Run the Backend

```bash
uvicorn main:app --reload
```

### 5. (Optional) Run the Streamlit UI

```bash
streamlit run frontend/streamlit_chat.py
```

---

## 🧠 Example Queries

| User Says                                  | Intent            |
|-------------------------------------------|-------------------|
| "Track my order"                           | Order tracking    |
| "Can I return this?"                       | Return policy     |
| "Suggest a dress under ₹1000"              | Recommendation    |
| "What's popular for weddings this month?"  | GPT-based reply   |

---

## 🧠 Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit or React
- **Vector Search:** FAISS
- **Embeddings:** Sentence Transformers
- **LLM:** OpenAI GPT-3.5 / Azure GPT
- **Database:** CSVs (or extend to SQL)
- **Deployment:** GCP / Render / Railway

---

## 💡 Next Goals

- [ ] Add authentication & session memory
- [ ] Plug into real product APIs (Amazon, Flipkart)
- [ ] Add multilingual support (e.g. Hindi queries)
- [ ] Add fallback human support option
- [ ] Deploy as a microservice with caching

---

## 🧾 License

MIT License

---

## 🙋‍♂️ Maintainer

Built by [Your Name] as a project to explore real-world AI and agentic assistant integration in e-commerce.

