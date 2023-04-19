# LTV API

## Setup
```
conda create -n 'ltv-api' python=3.9 -y
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
sudo apt install nginx -y
```

Create nginx config
```
sudo bash -c 'read -p "Enter the IP of your EC2 instance: " ec2_ip && cat > /etc/nginx/sites-enabled/fastapi_nginx << EOL
server {
    listen 80;
    server_name $ec2_ip;
    location / {
        proxy_pass http://127.0.0.1:8000;
        client_max_body_size 200M;
    }
}
EOL'
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