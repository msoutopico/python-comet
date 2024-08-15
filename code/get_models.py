import os

from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
os.environ["HUGGINGFACE_HUB_CACHE"] = (
    "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
)
# os.environ["TRANSFORMERS_CACHE"] = "/run/media/souto/257-FLASH/.cache/huggingface/"
os.environ["HF_DATASETS_CACHE"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
os.environ["HF_HOME"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"

custom_hf_cache_dpath = "/run/media/souto/257-FLASH/.cache/huggingface/hub"

model = "models--Unbabel--wmt20-comet-qe-da"


snapshot_download(
    repo_id="Unbabel/wmt20-comet-qe-da",
    repo_type="model",
    cache_dir=f"{custom_hf_cache_dpath}/{model}",
)
