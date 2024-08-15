import os

import comet_ml
from comet import download_model, load_from_checkpoint
from dotenv import load_dotenv
from rich import print

load_dotenv(override=True)


api_key = os.getenv("API_KEY")
experiment = comet_ml.Experiment(api_key=api_key, project_name="test")

print(f"{api_key=}")

# experiment = comet_ml.Experiment(api_key=api_key)


model_path = download_model("Unbabel/wmt22-cometkiwi-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

# Data must be in the following format:
data = [
    {
        "src": "The output signal provides constant sync so the display never glitches.",
        "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört.",
    },
    {
        "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
        "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років.",
    },
    {
        "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
        "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。",
    },
]
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)
print(model_output)

print(f"{model_output.scores=}")  # sentence-level scores
print(f"{model_output.system_score=}")  # system-level score

# Not all COMET models return metadata with detected errors.
if "metadata" in list(model_output.__dict__.keys()):
    print(f"{model_output.metadata.error_spans=}")  # detected error spans


# problems:
# 1. slow, model is updated every time
# 2. XCOMET-XL model does not load
# 4. questions:
#   - what is the system score?

# comet score specific to quality estimation that does not require reference trasnlations
# https://huggingface.co/Unbabel/wmt22-cometkiwi-da
# https://github.com/Unbabel/COMET/blob/v2.0.1/MODELS.md
# https://aclanthology.org/2022.wmt-1.60/
