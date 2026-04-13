from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import embedding


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str = ""
    groq_api_key: str = ""
    hf_token: str = ""
    open_router_api_key: str = ""

    pushover_user: str = ""
    pushover_token: str = ""
    pushover_url: str = "https://api.pushover.net/1/messages.json"

    ##rss_feed
    rss_feed_url: list[str] = [
        "https://www.dealnews.com/c142/Electronics/?rss=1",
        "https://www.dealnews.com/c39/Computers/?rss=1",
        "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    ]

    # agent_models
    scanner_model: str = "gpt-5-mini"
    messenger_model: str = "claude-sonnet-4-5"
    frontier_model: str = "gpt-4o-mini"
    frontier_reasoning_effort: str = "none"

    preprocessor_model: str = "groq/openai/gpt-oss-120b"
    preprocessor_reasoning_effort: str = "low"

    # ensemble_weights
    ensemble_frontier_weight: float = 0.8115124189175806
    ensemble_specialist_weight: float = 0.1884875810824194

    # embeddings
    embedding_model: str = "sentence-transformers/all-Mini-LM-L6-v2"

    # modal app(pricer.py)

    modal_app_name: str = "pricer"
    modal_gpu: str = "T4"
    base_llm: str = "meta-llama/Llama-3.1-8B-Instruct"

    finetuned_model: str = "Vishy08/product-pricer-08-12-2025_04.35.08"
    model_hf_secret: str = "huggingface-secret"
    modal_volume_name: str = "hf-hub-cache"


settings = Settings()
