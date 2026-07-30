from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Trusted Banking Regulatory RAG Web product")
    parser.add_argument(
        "--host",
        default=os.getenv("REG_RAG_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", os.getenv("REG_RAG_PORT", "7860"))),
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    uvicorn.run(
        "reg_rag.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
