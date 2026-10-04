"""Configuration constants and paths for Antigravity Hybrid Agent."""

import os

# Filesystem & Persistence Directories
SESSIONS_DIR = os.path.expanduser("~/.gemini/antigravity-cli/sessions")
MCP_CONFIG_PATH = os.path.expanduser("~/.gemini/config/mcp_config.json")
KONTEXTA_MCP_PATH = os.path.expanduser("~/Projects/kontexta/apps/mcp/dist/index.js")
KONTEXTA_DATA_DIR = os.path.expanduser("~/.local/share/kontexta")
DEFAULT_MODEL_DIR = "/home/safiyu/models"
OAUTH_TOKEN_PATH = os.path.expanduser("~/.gemini/antigravity-cli/antigravity-oauth-token")

# Default Service Endpoints & Ports
PORT_ROCM = 9000
PORT_CUDA = 9001
PORT_LLAMASHIFT = 8002
PORT_LAYA = 8003

DEFAULT_CLOUD_MODEL = "gemini-3.8-flash-high"
DEFAULT_LAYA_ENDPOINT = "http://localhost:8003/v1/systemone"
DEFAULT_LLAMASHIFT_URL = "http://localhost:8002"

# Endpoints Registry Initial Schema
DEFAULT_ENDPOINTS = {
    "9000": {
        "port": PORT_ROCM,
        "name": "ROCm:9000",
        "desc": "AMD ROCm GPU (Primary Executive)",
        "url": f"http://localhost:{PORT_ROCM}/v1",
        "model": "",
        "filename": "",
        "name_friendly": "",
        "status": "offline",
        "source": "none",
    },
    "9001": {
        "port": PORT_CUDA,
        "name": "CUDA:9001 (Compactor)",
        "desc": "NVIDIA RTX 4060 Context Synthesis Engine",
        "url": f"http://localhost:{PORT_CUDA}/v1",
        "model": "",
        "filename": "",
        "name_friendly": "",
        "status": "offline",
        "source": "none",
    },
}
