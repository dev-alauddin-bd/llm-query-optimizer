# 🤖 Production-Grade LLM Query Optimizer API using LangChain
A high-performance FastAPI/LangChain API utilizing Groq Cloud (Llama 3.3 70B) to sanitize, transliterate, and optimize input queries for improved Vector DB (RAG) performance.
---## ✨ Features* **FastAPI Backend:** Asynchronous `/api/v1/optimize` endpoint.* **Domain-Aware:** Contextually handles technical, medical, and legal terminology without hardcoding.
* **Deterministic Output:** Uses `temperature=0.0` for precise, non-hallucinatory results.
* **`uv` Workflow:** Modern, fast dependency management with `uv`.
---## 🏗️ Architecture & Workflow```text
Input -> FastAPI -> LangChain (Llama 3.3) -> Optimized Query -> JSON Response
```
---## 🛠️ Tech Stack* **Frameworks:** FastAPI, LangChain (`langchain-groq`, `langchain-core`)
* **LLM:** Groq Cloud (`llama-3.3-70b-versatile`)
* **Manager:** Astral `uv`* **Validation:** Pydantic
---## 🚀 Getting Started### 1. PrerequisitesPython 3.10+, `uv` installed.
### 2. Setup```bash
git clone <repository-url>
cd llm-query-optimizer
uv venv
uv add fastapi uvicorn langchain-groq langchain-core python-dotenv
```
### 3. Environment & RunningAdd `GROQ_API_KEY` to `.env`. Run with:```bash
.venv\Scripts\python.exe -m uvicorn main:app --reload
```
---## 🧪 Testing & API DocumentationNavigate to `http://127.0.0` to test the `POST` endpoint with raw Bengali/English mixed input [A, 1].
---## 📜 LicenseMIT License.


