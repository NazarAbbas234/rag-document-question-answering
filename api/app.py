from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile, os

from contract_qa_dashboard.ingest import ingest
from contract_qa_dashboard.query import query
from fastapi import Request

from pydantic import BaseModel
from typing import Optional

import logging
from api.schemas import QueryRequest, QueryResponse

# ----------------------------------------------------
# LOGGING CONFIG
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("contract_api")

# ----------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------
app = FastAPI(title="AI Contract Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------
# REQUEST LOGGING MIDDLEWARE
# ----------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log the incoming request
    body = await request.body()
    logger.info(f"Incoming request: {request.method} {request.url} - Body: {body.decode('utf-8')}")

    # Process the request
    response = await call_next(request)

    # Log the outgoing response
    logger.info(f"Response status: {response.status_code}")

    return response
# ----------------------------------------------------
# ROOT ENDPOINT
# ----------------------------------------------------
@app.get("/")
async def root():
    return {"message": "AI Contract Q&A API is running"}


# ----------------------------------------------------
# INGEST ENDPOINT
# ----------------------------------------------------
@app.post("/ingest")
async def ingest_docs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    logger.info(f"Received {len(files)} file(s) for ingestion.")

    try:
        temp_dir = tempfile.mkdtemp()
        paths = []

        for f in files:
            if f.filename == "":
                raise HTTPException(status_code=400, detail="One of the uploaded files has no filename.")

            file_path = os.path.join(temp_dir, f.filename)

            with open(file_path, "wb") as out:
                out.write(await f.read())

            paths.append(file_path)

        ingest(paths)

        logger.info("Ingestion completed.")
        return {"status": "success", "message": f"{len(paths)} document(s) ingested."}

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ----------------------------------------------------
# QUERY ENDPOINT
# ----------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):

    question = request.question.strip()
    model = request.model.lower()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Running query using model [{model}]: {question}")

    try:
        # ---- MODEL ROUTER (TEMP VERSION) ----
        if model == "openai":
            results = query([question], top_k=1)

        elif model == "ollama":
            results = query([question], top_k=1, llm_provider="ollama")

        elif model == "claude":
            results = query([question], top_k=1, llm_provider="claude")

        else:
            raise HTTPException(status_code=400, detail="Invalid model selected.")


        result = results[0]

        return QueryResponse(
            summary=result["summary"],
            clause_type=result["clause_type"],
            risk_level=result["risk_level"],
            explanation=result["explanation"],
            model_used=model
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No index found. Please ingest documents first."
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

