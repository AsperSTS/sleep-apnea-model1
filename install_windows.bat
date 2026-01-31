@echo off
call conda env create -f environment.yml
call conda activate python311_psg
python -m spacy download en_core_web_sm
pause