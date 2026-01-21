# Green IT ROI Dashboard (LVMH France)

Streamlit app to compare **New vs Refurbished** IT devices using:
- Financial impact (TCO / annualized cost)
- Environmental impact (CO₂e)
- Fleet volumetry by device type
- Full parameter documentation with sources & assumptions

### DESCRIPTION DES FICHIERS

# COMMENT RUN LE CODE

## Project structure
- `app/` : Streamlit app entry + pages
- `src/` : business logic (calculations, plotting, documentation)
- `data/` : sample/raw data (optional)
- `assets/` : images, styles
- `scripts/` : helper scripts (optional)

## Setup
### 1) Create a virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
