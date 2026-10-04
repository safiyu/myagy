"""Llama-Shift & Llama-Server discovery, probing, and dynamic model hot-swap."""

import os
import json
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any

from ..config import DEFAULT_MODEL_DIR, DEFAULT_LLAMASHIFT_URL


def detect_port_model(port: int | str, fallback_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Dynamically detects the currently active model on a given port.
    Never hardcodes filenames. Seamlessly adapts when llama-shift switches models.
    Priority:
      1. GET /api/active  — dedicated endpoint, returns only running models (fastest).
      2. GET /api/status  — filters models[] where status == 'running' for this port.
      3. GET /v1/models   — direct llama-server query as last resort.
    """
    port_int = int(port)
    model_dir = DEFAULT_MODEL_DIR

    def _build_result(m: Dict, source: str) -> Dict[str, Any]:
        fname = m.get("filename") or ""
        full_p = os.path.join(model_dir, fname) if fname and os.path.exists(os.path.join(model_dir, fname)) else fname
        return {
            "model_path": full_p or fname,
            "filename": fname or os.path.basename(full_p),
            "name": m.get("name") or fname,
            "id": m.get("id", ""),
            "status": "online",
            "source": source,
            "port": port_int,
            "gpu": m.get("gpu", ""),
        }

    def _port_matches(m: Dict) -> bool:
        return m.get("port") == port_int or f":{port_int}" in m.get("endpoint", "")

    # 1. /api/active — only running models, single call
    try:
        req = urllib.request.Request(
            f"{DEFAULT_LLAMASHIFT_URL}/api/active",
            headers={"User-Agent": "Antigravity/1.0"}
        )
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            active = json.loads(resp.read().decode("utf-8"))
            if isinstance(active, list):
                for m in active:
                    if _port_matches(m):
                        return _build_result(m, "llamashift-active")
    except Exception:
        pass

    # 2. /api/status — full list, filter to running + matching port
    try:
        req = urllib.request.Request(
            f"{DEFAULT_LLAMASHIFT_URL}/api/status",
            headers={"User-Agent": "Antigravity/1.0"}
        )
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                if _port_matches(m) and m.get("status") == "running":
                    return _build_result(m, "llamashift")
    except Exception:
        pass

    # 3. Direct llama-server /v1/models query
    url = fallback_url or f"http://localhost:{port_int}/v1"
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/models",
            headers={"User-Agent": "Antigravity/1.0"}
        )
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            model_id = None
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                model_id = data["data"][0].get("id") or data["data"][0].get("name")
            elif "models" in data and isinstance(data["models"], list) and len(data["models"]) > 0:
                model_id = data["models"][0].get("name") or data["models"][0].get("model")
            if model_id:
                fname = os.path.basename(model_id)
                return {
                    "model_path": model_id,
                    "filename": fname,
                    "name": fname,
                    "id": fname,
                    "status": "online",
                    "source": "llama-server",
                    "port": port_int,
                    "gpu": "",
                }
    except Exception:
        pass

    return {
        "model_path": "",
        "filename": "[Offline / No Model Loaded]",
        "name": "Offline",
        "id": "",
        "status": "offline",
        "source": "none",
        "port": port_int,
        "gpu": "",
    }


def query_endpoint_model(url: str, timeout: float = 1.0) -> Optional[str]:
    """Compatibility wrapper that queries a model URL."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/models", headers={"User-Agent": "Antigravity/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                return data["data"][0].get("id") or data["data"][0].get("name")
            if "models" in data and isinstance(data["models"], list) and len(data["models"]) > 0:
                return data["models"][0].get("name") or data["models"][0].get("model")
    except Exception:
        pass
    return None


def trigger_llamashift_switch(model_id: str) -> Dict[str, Any]:
    """Triggers an in-GPU model switch or start via llama-shift HTTP API."""
    try:
        payload = json.dumps({"model": model_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{DEFAULT_LLAMASHIFT_URL}/api/start",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Antigravity/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def get_llamashift_active() -> List[Dict[str, Any]]:
    """
    Fetches only currently running models via GET /api/active.
    Returns a list of active model dicts.
    """
    try:
        req = urllib.request.Request(f"{DEFAULT_LLAMASHIFT_URL}/api/active", headers={"User-Agent": "Antigravity/1.0"})
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def get_llamashift_models() -> List[Dict[str, Any]]:
    """Fetches all configured model profiles from llama-shift /api/status (running + stopped)."""
    try:
        req = urllib.request.Request(f"{DEFAULT_LLAMASHIFT_URL}/api/status", headers={"User-Agent": "Antigravity/1.0"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "models" in data:
                mode = data.get("mode", "single_port")
                master_port = str(data.get("masterPort", 9000))
                models_list = []
                for m in data["models"]:
                    is_dedicated = bool(m.get("dedicated", False))
                    if mode == "single_port" and not is_dedicated:
                        effective_port = master_port
                    else:
                        effective_port = str(m.get("port", master_port))

                    models_list.append({
                        "id": m.get("id", ""),
                        "name": m.get("name", ""),
                        "filename": m.get("filename", ""),
                        "port": effective_port,
                        "raw_port": str(m.get("port", "")),
                        "dedicated": is_dedicated,
                        "status": m.get("status", "stopped"),
                        "gpu": m.get("gpu", ""),
                        "devices": m.get("devices", []),
                        "size": m.get("size", ""),
                        "desc": m.get("desc", ""),
                    })
                return models_list
    except Exception:
        pass
    return []
