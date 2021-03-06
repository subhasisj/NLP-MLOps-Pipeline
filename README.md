# NLP-MLOps-Pipeline
End to end MLOps Pipeline for NLP

- `Model`: Intent Detection Model Training using Huggingface Trainer API with `wandb` and `Hydra` integration

    To train a new model
    ```
    python train.py training.max_epochs=4 

    ```


- `DVC`: DVC integration for Model Versioning

    To push a new model to remote
    ```
    cd dvcfiles
    dvc add ../Notebooks/models/distilroberta-base/ --file clinc_distilroberta.dvc
    dvc push clinc_distilroberta.dvc
    ```


- `Model Compression`: ONNX / Quantization / Pruning

