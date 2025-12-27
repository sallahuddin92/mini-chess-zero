"""
Stock Predictor Web App
========================
Upload CSV, get instant prediction.
"""

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fast_predictor import FastPredictor

app = FastAPI(title="🇲🇾 Malaysia Stock Predictor")

# Load model once
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    predictor = FastPredictor()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🇲🇾 Malaysia Stock Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
            padding: 40px 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 40px;
        }
        .upload-box {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-box:hover {
            border-color: #00d4ff;
            background: rgba(0,212,255,0.1);
        }
        .upload-box.dragover {
            border-color: #00ff88;
            background: rgba(0,255,136,0.1);
        }
        input[type="file"] { display: none; }
        .upload-icon { font-size: 4rem; margin-bottom: 20px; }
        .upload-text { font-size: 1.2rem; color: #ccc; }
        .result {
            margin-top: 40px;
            background: rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 30px;
            display: none;
        }
        .result.show { display: block; animation: fadeIn 0.5s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .signal {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            margin: 20px 0;
        }
        .signal.buy { color: #00ff88; }
        .signal.sell { color: #ff4757; }
        .signal.hold { color: #ffa502; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        .metric {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #00d4ff;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #888;
            margin-top: 5px;
        }
        .stock-name {
            text-align: center;
            font-size: 1.5rem;
            color: #ccc;
            margin-bottom: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #00d4ff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .footer {
            text-align: center;
            margin-top: 60px;
            color: #555;
            font-size: 0.9rem;
        }
        .footer a { color: #00d4ff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🇲🇾 Malaysia Stock Predictor</h1>
        <p class="subtitle">Powered by Asymmetric TD Learning • Upload CSV, Get Instant Prediction</p>
        
        <div class="upload-box" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📊</div>
            <div class="upload-text">Drop stock CSV here or click to upload</div>
            <input type="file" id="fileInput" accept=".csv">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Analyzing stock data...</div>
        </div>
        
        <div class="result" id="result">
            <div class="stock-name" id="stockName"></div>
            <div class="signal" id="signalText"></div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="confidence">-</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="price">-</div>
                    <div class="metric-label">Current Price</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="momentum">-</div>
                    <div class="metric-label">5D Momentum</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Built with <a href="https://github.com/sallahuddin92/mini-chess-zero">Asymmetric TD Learning</a> • 
            © 2025 StarEmporium Enterprise
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) uploadFile(file);
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) uploadFile(e.target.files[0]);
        });
        
        async function uploadFile(file) {
            loading.classList.add('show');
            result.classList.remove('show');
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                loading.classList.remove('show');
                
                document.getElementById('stockName').textContent = file.name.replace('.csv', '');
                document.getElementById('signalText').textContent = data.signal;
                document.getElementById('signalText').className = 'signal ' + data.signal.toLowerCase();
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(0) + '%';
                document.getElementById('price').textContent = 'RM ' + data.price.toFixed(2);
                document.getElementById('momentum').textContent = data.momentum_5d.toFixed(1) + '%';
                
                result.classList.add('show');
            } catch (error) {
                loading.classList.remove('show');
                alert('Error: ' + error.message);
            }
        }
    </script>
</body>
</html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict stock signal from uploaded CSV."""
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Handle Investing.com format
        if 'Price' in df.columns:
            # Remove commas and convert to float
            df['Close'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        elif 'Close' in df.columns:
            df['Close'] = df['Close'].astype(str).str.replace(',', '').astype(float)
        else:
            # Find first numeric column
            for col in df.columns[1:]:  # Skip date column
                try:
                    df['Close'] = pd.to_numeric(df[col].astype(str).str.replace(',', ''))
                    break
                except:
                    continue
        
        # Reverse to chronological order and get prices
        prices = df['Close'].dropna().values[::-1].astype(np.float32)
        
        if len(prices) < 25:
            return {"error": "Need at least 25 data points", "signal": "ERROR", "confidence": 0, "price": 0, "momentum_5d": 0}
        
        result = predictor.predict(prices)
        return result
    except Exception as e:
        return {"error": str(e), "signal": "ERROR", "confidence": 0, "price": 0, "momentum_5d": 0}


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Stock Predictor Web App...")
    print("📊 Open http://localhost:8001 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
