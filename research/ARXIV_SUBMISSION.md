# arXiv Submission Guide

## Paper Details
- **Title**: Asymmetric Temporal Difference Learning: Biologically-Inspired Stabilization for Deep Reinforcement Learning
- **Authors**: StarEmporium Enterprise
- **Category**: cs.LG (Machine Learning) or cs.AI (Artificial Intelligence)

## Ready Files
| File | Purpose |
|------|---------|
| `research/paper/paper_draft.md` | Main paper (convert to LaTeX/PDF) |
| `research/figures/` | Publication figures |
| `README.md` | Project overview |

## Submission Steps

### Step 1: Create arXiv Account
1. Go to https://arxiv.org/
2. Click "Register" if you don't have an account
3. Verify email

### Step 2: Convert Paper to PDF
Option A: Use Pandoc
```bash
cd /Users/sallahuddin/mini_chess_rl/research/paper
pandoc paper_draft.md -o paper.pdf
```

Option B: Use online converter
- https://markdowntopdf.com/
- Paste content, download PDF

### Step 3: Submit
1. Login to arXiv
2. Click "Submit" → "New Submission"
3. Choose category: **cs.LG** (Machine Learning)
4. Upload PDF
5. Fill metadata:
   - Title: (from paper)
   - Authors: StarEmporium Enterprise
   - Abstract: (copy from paper_draft.md)
6. Submit for moderation

## Key Results to Highlight
- Win rate: 3% → **87.5%**
- Q-value reduction: 70M → **9.2** (1,380× reduction)
- Training games: 2,000

## Alternative: GitHub + README
If arXiv moderation takes too long, the GitHub repo already serves as publication:
https://github.com/sallahuddin92/mini-chess-zero
