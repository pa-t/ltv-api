# LTV API

## Setup
```
conda create -n 'ltv-api' python=3.9
conda activate ltv-api
pip3 install -r requirements.txt
```

## Running API
```
uvicorn main:app --reload --log-level info
```

## Setting Up EC2
- EC2 must have HTTP access from external IPs
- Clone the repo and follow the above steps

Install nginx
```
sudo apt install nginx
```

Create nginx config
```
sudo vi /etc/nginx/sites-enabled/fastapi_nginx
```

Paste the following into the file
```
server {
    listen 80;
    server_name <IP OF YOUR EC2>;
    location / {
        proxy_pass http://127.0.0.1:8000;
        client_max_body_size 200M;
    }
}
```

Restart nginx
```
sudo service nginx restart
```

Run the uvicorn command
```
uvicorn main:app --reload --log-level info &> ./app.log &
```

Visit the Public IPv4 address of your EC2 instance, for example:
```
12.345.567.89/docs
```