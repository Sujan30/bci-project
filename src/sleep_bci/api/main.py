"""
Entry point for the Sleep BCI API server.

Run with: sleepbci-serve
Or directly: python -m sleep_bci.api.main
"""
import os
import uvicorn
from sleep_bci.api.app import app


def main():
    """Start the FastAPI server."""
    # Read configuration from environment variables
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()