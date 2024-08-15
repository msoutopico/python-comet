# MT Quality Estimation

COMET models are trained to predict quality scores for translations. 

This is a very rough draft (just a few notes lumped together).

model Unbabel/wmt22-cometkiwi-da requires login to hf

`huggingface-cli login --token $HUGGINGFACE_TOKEN`


## Warnings

/run/media/souto/257-FLASH/python-comet/venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
Encoder model frozen.
/run/media/souto/257-FLASH/python-comet/venv/lib/python3.12/site-packages/pytorch_lightning/core/saving.py:195: Found keys that are not in the model state dict but in the checkpoint: ['encoder.model.embeddings.position_ids']




/run/media/souto/257-FLASH/python-comet/venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:419: Consider setting `persistent_workers=True` in 'predict_dataloader' to speed up the dataloader worker initialization.
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.63it/s]

## Test 

curl --location --request GET 'http://127.0.0.1:8000/api/scores' \
--header 'Content-Type: application/json' \
--data '[
    {"src": "How to Demonstrate Your Strategic Thinking Skills", "mt": "Cómo demostrar su capacidad de pensamiento estratégico" },{ "src": "Why is Accuracy important in the workplace?", "mt": "¿Por qué es importante la precisión en el trabajo" }, { "src": "When faced with a large amount of analysis ask for support setting up a team to approach the issue in different ways.", "mt": "Cuando se enfrente a una gran cantidad de análisis, pida ayuda para crear un equipo que aborde la cuestión de diferentes maneras." }
]'


## expected output: 

{
    "data": [
        {
            "src": "How to Demonstrate Your Strategic Thinking Skills",
            "mt": "Cómo demostrar su capacidad de pensamiento estratégico",
            "score": 0.5087478756904602
        },
        {
            "src": "Why is Accuracy important in the workplace?",
            "mt": "¿Por qué es importante la precisión en el trabajo",
            "score": 0.6997313499450684
        },
        {
            "src": "When faced with a large amount of analysis ask for support setting up a team to approach the issue in different ways.",
            "mt": "Cuando se enfrente a una gran cantidad de análisis, pida ayuda para crear un equipo que aborde la cuestión de diferentes maneras.",
            "score": 0.41031646728515625
        }
    ],
    "model_output_system_score": 0.5395985643068949
}
## References 

https://github.com/Unbabel/COMET


## Caveats

auth

Lightning automatically upgraded your loaded checkpoint from v1.8.2 to v2.4.0. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt`


python -m pytorch_lightning.utilities.upgrade_checkpoint ../.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt

## steps

dev / testing

- go to python-comet's root folder
- login to hf
- start api

deployment / production

- install in machine with gpu or tpu
- unicorn. etc. get url

if deployed to the world, some basic authentication is necessary

run omtmt4pe

