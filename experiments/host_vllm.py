"""
vLLM server hosting script for Gemma-2-9b with LoRA adapters.

Usage:
    python host_vllm.py

This will start an OpenAI-compatible API server on http://localhost:8000
You can then query it using the OpenAI API format with specific LoRA adapters.
"""

import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import run_server

MODEL_NAME = "google/gemma-2-9b-it"
MAX_MODEL_LEN = 4096
PORT = 8000
HOST = "0.0.0.0"

LORA_MODULES_LIST = [
    "thejaminator/extreme_sports-gemma-2-9b-it-sft-20251029",
    "thejaminator/bad_medical-gemma-2-9b-it-sft-20251029",
    "thejaminator/riskymix-gemma-2-9b-it-sft-20251029",
]

LORA_MODULES = {name: name for name in LORA_MODULES_LIST}


def main():
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        enable_lora=True,
        max_lora_rank=32,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    print(f"Starting vLLM server with model: {MODEL_NAME}")
    print(f"Server will be available at: http://{HOST}:{PORT}")
    print(f"LoRA adapters available: {list(LORA_MODULES.keys())}")
    print(
        "\nTo use a specific LoRA adapter, set 'model' to the full model name in your API request"
    )
    print("\nPress Ctrl+C to stop the server\n")

    asyncio.run(
        run_server(
            engine_args,
            host=HOST,
            port=PORT,
            lora_modules=LORA_MODULES,
        )
    )


if __name__ == "__main__":
    main()
