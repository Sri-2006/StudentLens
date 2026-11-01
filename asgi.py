"""ASGI entry point for Vercel and other ASGI-compatible Python hosts.

Keep this separate from app.py: importing the deployment entry point must not
execute the Streamlit page outside a browser session.
"""
from pathlib import Path

import streamlit as st
from starlette.responses import JSONResponse
from starlette.routing import Route


async def health(_request):
    return JSONResponse({"status": "ok", "app": "StudentLens"})


app = st.App(
    Path(__file__).resolve().with_name("app.py"),
    routes=[Route("/healthz", health)],
)
