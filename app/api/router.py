from fastapi import APIRouter
from app.api.endpoints import issues

api_router = APIRouter()
api_router.include_router(issues.router, prefix="/issues", tags=["issues"])
