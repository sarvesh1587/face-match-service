"""API Router — aggregates all sub-routers."""

from fastapi import APIRouter

from app.api.endpoints import enroll, health, metrics, search

api_router = APIRouter()
api_router.include_router(enroll.router, tags=["Enrollment"])
api_router.include_router(search.router, tags=["Search"])
api_router.include_router(health.router, tags=["Operations"])
api_router.include_router(metrics.router, tags=["Operations"])
