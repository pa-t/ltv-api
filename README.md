# LTV API

## Setup
```
conda create -n 'ltv-api' python=3.9
conda activate ltv-api
pip3 install -r requirements.txt
```

## AWS Setup
Need the following ENV VARS for AWS to talk to services:
```
AWS_ACCESS_KEY_ID=<KEY>
AWS_SECRET_ACCESS_KEY=<SECRET>
AWS_DEFAULT_REGION=us-east-1
```

## Running API locally
```
uvicorn main:app --reload
```
