import uvicorn
from src.utils import load_config

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    api_cfg = config["api"]

    uvicorn.run(
        "src.api.app:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        workers=1,
        log_level="info",
        reload=False,
    )