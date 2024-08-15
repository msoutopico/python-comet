import os

from comet import load_from_checkpoint

from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
os.environ["HUGGINGFACE_HUB_CACHE"] = (
    "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
)
# os.environ["TRANSFORMERS_CACHE"] = "/run/media/souto/257-FLASH/.cache/huggingface/"
os.environ["HF_DATASETS_CACHE"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"
os.environ["HF_HOME"] = "/run/media/souto/257-FLASH/.cache/huggingface/hub/"

custom_hf_cache_dpath = "/run/media/souto/257-FLASH/.cache/huggingface/hub"
org = "Unbabel"
model = "wmt20-comet-qe-da"
# model = "wmt22-cometkiwi-da"
folder = f"models--{org}--{model}"
repo_id = f"{org}/{model}"

clean_up_tokenization_spaces = True

# from comet import download_model, load_from_checkpoint
# model_path = download_model(f"{org}/{model}")

# snapshot_download(
#     repo_id=f"{org}/{model}",
#     repo_type="model",
#     cache_dir=f"{custom_hf_cache_dpath}/{folder}",
# )

if repo_id == "Unbabel/wmt22-cometkiwi-da":
    model_path = f"{custom_hf_cache_dpath}/models--{org}--{model}/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt"
elif repo_id == "Unbabel/wmt20-comet-qe-da":
    model_path = f"{custom_hf_cache_dpath}/models--{org}--{model}s/snapshots/2e7ffc84fb67d99cf92506611766463bb9230cfb/checkpoints/model.ckpt"

model = load_from_checkpoint(model_path)

data = [
    {
        "src": "How to Demonstrate Your Strategic Thinking Skills",
        "mt": "Cómo demostrar su capacidad de pensamiento estratégico",
    },
    {
        "src": "Why is Accuracy important in the workplace?",
        "mt": "¿Por qué es importante la precisión en el trabajo",
    },
    {
        "src": "When faced with a large amount of analysis ask for support setting up a team to approach the issue in different ways.",
        "mt": "Cuando se enfrente a una gran cantidad de análisis, pida ayuda para crear un equipo que aborde la cuestión de diferentes maneras.",
    },
]


def produce_scores(data):
    # model_output = model.predict(data, batch_size=8, gpus=1)
    # print(f"{model_output.scores=}")  # sentence-level scores
    # print(f"{model_output.system_score=}")  # average score
    return model.predict(data, batch_size=8, gpus=1).scores


x = produce_scores(data)
print(x)
