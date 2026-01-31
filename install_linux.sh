#!/bin/bash
conda env create -f environment.yml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate python311_psg
python -m spacy download en_core_web_sm