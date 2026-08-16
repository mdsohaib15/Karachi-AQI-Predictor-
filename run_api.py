"""
Run FastAPI Backend Server
==========================
Launches the FastAPI application on http://localhost:8000
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI Backend Server on http://localhost:8000 ...")
    print("Interactive Swagger Documentation: http://localhost:8000/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
