"""Public FastAPI entry point for the Tally AI service."""

from fastapi import FastAPI

from backend.api import app

app: FastAPI

__all__ = ["app"]