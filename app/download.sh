#!/bin/bash
mkdir models
curl https://zenodo.org/record/4599830/files/roberta-large.zip?download=1 --output models/roberta-large.zip
unzip models/roberta-large.zip -d models/
rm models/roberta-large.zip