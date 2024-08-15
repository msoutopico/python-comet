import os
from code.models.mtpe import ScoredTranslation, Translation
from typing import List

from comet import load_from_checkpoint
from dotenv import load_dotenv
from fastapi import APIRouter  # HTTPException, status
from huggingface_hub import login

# from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
os.environ["HUGGINGFACE_HUB_CACHE"] = (
    "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
)
# os.environ["TRANSFORMERS_CACHE"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
os.environ["HF_DATASETS_CACHE"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
os.environ["HF_HOME"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"

load_dotenv()
TOKEN = os.environ["HUGGINGFACE_TOKEN"]
login(TOKEN)

custom_hf_cache_dpath = "/run/media/souto/257-FLASH/.cache/huggingface/hub"
org = "Unbabel"
# model = "wmt20-comet-qe-da"
model = "wmt22-cometkiwi-da"
folder = f"models--{org}--{model}"
repo_id = f"{org}/{model}"

clean_up_tokenization_spaces = True


router = APIRouter()

# from comet import download_model, load_from_checkpoint
# model_path = download_model(f"{org}/{model}")

# snapshot_download(
#     repo_id="Unbabel/wmt20-comet-qe-da",
#     repo_type="model",
#     cache_dir=f"{custom_hf_cache_dpath}/models--{org}--{model}",
# )

if repo_id == "Unbabel/wmt22-cometkiwi-da":
    model_path = f"{custom_hf_cache_dpath}/models--{org}--{model}/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt"
elif repo_id == "Unbabel/wmt20-comet-qe-da":
    model_path = f"{custom_hf_cache_dpath}/models--{org}--{model}s/snapshots/2e7ffc84fb67d99cf92506611766463bb9230cfb/checkpoints/model.ckpt"

model = load_from_checkpoint(model_path)


def produce_scores(data):
    data_dict = [{"src": obj.src, "mt": obj.mt} for obj in data]
    # model_output = model.predict(data, batch_size=8, gpus=1)
    # print(f"{model_output.scores=}")  # sentence-level scores
    # print(f"{model_output.system_score=}")  # average score
    return model.predict(data_dict, batch_size=8, gpus=1).scores


def add_scores_to_data(data, scores):
    # scores = produce_scores(data)
    return [
        ScoredTranslation(
            **translation.dict(),  # Unpack the Translation instance data
            score=score,  # Add the new score value
        )
        for translation, score in zip(data, scores)
    ]


@router.get("/scores")
async def get_scores(translations: List[Translation]):
    # model_output_scores = [0.3048415184020996, 0.23436091840267181, 0.6128204464912415]
    # model_output_system_score = 0.38400762776533764

    scores = produce_scores(translations)
    return {
        "data": add_scores_to_data(translations, scores),
        "model_output_system_score": sum(scores) / len(scores),
    }
