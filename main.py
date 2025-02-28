from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import crm_email_api

# Initialize FastAPI app
app = FastAPI(title="Email Generation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(crm_email_api.router)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Email Generation API"}