# Training Stocks Folder
Place your stock CSV files here for multi-stock training.

## Recommended Stocks (10-20 for best results)

Download from Investing.com for each stock:

### Blue Chips (Essential - 5 stocks)
1. MAYBANK - Malayan Banking
2. CIMB - CIMB Group
3. PBBANK - Public Bank
4. TENAGA - Tenaga Nasional
5. PCHEM - Petronas Chemicals

### Tech/Growth (4 stocks)
6. INARI - Inari Amertron
7. TM - Telekom Malaysia
8. AXIATA - Axiata Group
9. VITROX - ViTrox Corporation

### Healthcare/Consumer (3 stocks)
10. IHH - IHH Healthcare
11. NESTLE - Nestle Malaysia
12. PPB - PPB Group

### Diversified (3 stocks)
13. SIME - Sime Darby
14. MISC - MISC Berhad
15. GENTING - Genting

## How to Download

1. Go to: https://investing.com/equities/
2. Search for stock (e.g., "MAYBANK")
3. Click "Historical Data"
4. Set period: Max or 5 Years
5. Download CSV
6. Save to this folder

## File Format

Files should have columns: Date, Price, Open, High, Low, Vol., Change %
(Investing.com format works directly)

## After Adding Files

Run training:
```bash
cd /Users/sallahuddin/mini_chess_rl/trading
python -c "from fast_predictor import FastPredictor; p = FastPredictor(); p.train_multi_stock('training_stocks')"
```
