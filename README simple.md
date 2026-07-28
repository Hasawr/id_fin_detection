# Linux RTX 5090 — quick reference (stable + public subdomain)

Public API: **https://ocr-fin.biletim.az**  
Local API: `http://127.0.0.1:8010` (uvicorn / systemd only)

## 1. Production API (systemd)

```bash
cd ~/Desktop/OCRAsService/id_fin_detection-feture

cp .env.production.example .env
nano .env   # set a real API_KEYS value (32+ chars)

sudo cp deploy/ocr-api.service /etc/systemd/system/ocr-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-api

curl http://127.0.0.1:8010/health
sudo systemctl status ocr-api
```

Always use **`.venv-gpu`**. Never start production from the old `.venv`.

## 2. HTTPS for ocr-fin.biletim.az (nginx)

DNS for `ocr-fin.biletim.az` must already point at this server.

```bash
# install nginx + certbot if missing
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx

# install site config
sudo cp deploy/nginx-ocr-fin.biletim.az.conf \
  /etc/nginx/sites-available/ocr-fin.biletim.az
sudo ln -sf /etc/nginx/sites-available/ocr-fin.biletim.az \
  /etc/nginx/sites-enabled/ocr-fin.biletim.az

# first enable HTTP only so certbot can run, if SSL paths do not exist yet:
# temporarily comment the whole "listen 443" server block, then:
sudo nginx -t && sudo systemctl reload nginx

# issue certificate (opens/uses port 80)
sudo certbot --nginx -d ocr-fin.biletim.az

# if network team already gave cert files, skip certbot and set
# ssl_certificate / ssl_certificate_key in the nginx config to their paths.

sudo nginx -t && sudo systemctl reload nginx
```

Check:

```bash
curl https://ocr-fin.biletim.az/health
# with API key:
curl -X POST "https://ocr-fin.biletim.az/v1/id-fin" \
  -H "X-API-Key: YOUR_KEY" \
  -F "mrz=@/path/to/id-back.jpg"
```

Do **not** put Streamlit behind this subdomain.

## 3. Manual API start (debug only)

```bash
cd ~/Desktop/OCRAsService/id_fin_detection-feture
source .venv-gpu/bin/activate
bash scripts/start_api_linux.sh
```

## 4. Internal Streamlit demo (loopback only)

```bash
cd ~/Desktop/OCRAsService/id_fin_detection-feture
source .venv-gpu/bin/activate
# OCR_API_BASE_URL=http://127.0.0.1:8010
streamlit run demos/streamlit_app.py --server.address 127.0.0.1 --server.port 8510
```

Open http://127.0.0.1:8510
