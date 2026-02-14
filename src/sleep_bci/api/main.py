"""
Entry point for the Sleep BCI API server.

Run with: sleepbci-serve
Or directly: python -m sleep_bci.api.main
"""
import uvicorn
from sleep_bci.api.app import app


def main():
    """Start the FastAPI server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()