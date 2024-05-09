
import os
from rich import print

import comet_ml

from dataclasses import dataclass
from typing import Optional

import comet
import datasets
import torch
import evaluate

from comet import download_model, load_from_checkpoint

from dotenv import load_dotenv
load_dotenv(override=True)


api_key = os.getenv("API_KEY")
experiment = comet_ml.Experiment(
    api_key=api_key,
    project_name="test"
)

# experiment = comet_ml.Experiment(api_key=api_key)



# Choose your model from Hugging Face Hub
# model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)


# Data must be in the following format:
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    }
]

print(data)
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)

print(f"{model_output.scores=}") # sentence-level scores
print(f"{model_output.system_score=}") # system-level score

# Not all COMET models return metadata with detected errors.
if "metadata" in list(model_output.__dict__.keys()):
    print(f"{model_output.metadata.error_spans=}") # detected error spans







# problems:
# 1. slow, model is updated every time
# 2. XCOMET-XL model does not load
# 3. ref?
# 4. questions:
#   - what is the system score?

# comet score specific to quality estimation that does not require reference trasnlations
# https://huggingface.co/Unbabel/wmt22-cometkiwi-da
# https://github.com/Unbabel/COMET/blob/v2.0.1/MODELS.md
# https://aclanthology.org/2022.wmt-1.60/
