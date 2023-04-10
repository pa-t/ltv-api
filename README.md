# LTV API

## Setup
```
conda create -n 'ltv-api' python=3.9
conda activate ltv-api
pip3 install -r requirements.txt
```

## Running API locally
```
uvicorn main:app --reload --log-level info
```
