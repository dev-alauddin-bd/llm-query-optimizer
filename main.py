import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from the security-configured .env file
load_dotenv()

# Initialize the production-ready FastAPI application configuration
app = FastAPI(
    title="LLM Query Optimizer API",
    description="Production-grade API layer that utilizes LangChain and Llama 3.3 70B to contextually rewrite unpolished queries.",
    version="1.0.0"
)

# Global guardrail verifying API keys exist before exposing endpoints
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("⚠️ Critical Error: GROQ_API_KEY could not be located in your local .env file!")

# Initialize the core model engine at absolute deterministic output parsing
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

# Verified universal, domain-agnostic query rewriter prompt template
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly capable, multilingual query optimizer. Your sole task is to analyze the user's input—which may be written in "Banglish" (Bengali phonetics transliterated into the Roman/Latin alphabet)—and rewrite it into a highly professional, grammatically perfect English query optimized for LLM execution.

    Core Directives:
    1. **Contextual Transliteration:** Intelligently map Romanized Bengali phonetics into their true semantic English definitions based on the global context of the sentence (e.g., distinguish between phonetic overlaps and actual English words).
    2. **Preserve Intent:** Keep the exact core constraints, intent, and domain focus of the user's question intact.
    3. **No Direct Answering:** Do not answer, explain, or interpret the question yourself.
    4. **Strict Output Control:** Output ONLY the final rewritten question string in plain English. Absolutely no preamble, intros, outros, or markdown formatting."""),
    ("human", "{question}")
])

# Assemble the declarative LangChain runtime chain execution pipeline
rewriter_chain = rewrite_prompt | llm | StrOutputParser()


# Define data schemas for API requests and JSON responses using Pydantic Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The raw, unpolished, or Banglish query string sent from the client application.")

class QueryResponse(BaseModel):
    status: str
    original_query: str
    optimized_query: str
    ai_response: str


# 🚀 Production API Post Endpoint
@app.post(
    "/api/v1/optimize", 
    response_model=QueryResponse, 
    status_code=status.HTTP_200_OK,
    summary="Optimize and execute raw unpolished user inputs."
)
async def process_and_optimize_query(request_data: QueryRequest):
    """
    Accepts raw text query payload, intercepts it at the gateway to fix grammar and spelling ambiguities, 
    dispatches the formal rewritten string to the model layer, and returns a clean, structured JSON response.
    """
    if not request_data.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input question string cannot be empty.")
        
    try:
        # Step 1: Execute the optimization chain pipeline to rewrite inputs contextually
        optimized_query_string = rewriter_chain.invoke({"question": request_data.question})
        
        # Step 2: Forward the perfectly formatted output string directly to the final core execution model 
        final_model_execution = llm.invoke(optimized_query_string)
        
        # Return structured JSON payload perfectly built for frontend processing integrations
        return QueryResponse(
            status="success",
            original_query=request_data.question,
            optimized_query=optimized_query_string,
            ai_response=final_model_execution.content
        )
    except Exception as error_exception:
        # Secure server-side error capturing schema configurations
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"An infrastructure execution fault occurred: {str(error_exception)}"
        )
