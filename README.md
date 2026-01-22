# Green IT ROI Dashboard — New vs Refurbished

A **decision-support tool** built with Streamlit to compare **new vs refurbished IT equipment** using:


### FILES DESCRIPTION

# HOW TO RUN THE CODE

## Project structure
- `app/` : Streamlit app entry + pages
- `src/` : business logic (calculations, plotting, documentation)
- `data/` : sample/raw data (optional)
- `assets/` : images, styles
- `scripts/` : helper scripts (optional)

- Financial performance (TCO, annualized cost)
- Environmental impact (kgCO₂e)
- Fleet volumetry (number of devices by type)
- Audit-ready documentation of all assumptions and sources

Designed for **corporate IT, procurement, and sustainability teams** (e.g. LVMH France).

---

## 1. Project Structure

LVMH/
├─ app.py
├─ requirements.txt
├─ environment.yml
├─ README.md
├─ data/
└─ .venv/ # local virtual environment (not committed)


---

## 2. Dependencies

The application uses the following Python libraries:

- streamlit
- numpy
- pandas
- matplotlib

All dependencies are defined in:
- requirements.txt
- environment.yml

---

## 3. Installation

From the project root (the folder containing `app.py`):

### Option A — pip + virtual environment


**Windows (PowerShell)**

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt


---

## 4. Running the Application

From the project root:

python -m streamlit run app.py


On Windows, always use `python -m streamlit` instead of `streamlit run`.

---

## 5. How to Use the App

### 5.1 Select a Device Type

In the sidebar:
- Choose Laptop, Tablet, Smartphone, Switch / Router, or Room Screens
- If Smartphone is selected, the brand is automatically restricted to Apple

### 5.2 Enter Device Parameters

For one device:
- Purchase price (€)
- Production + transport CO₂ footprint (kgCO₂e)
- Theoretical lifetime (years)
- Annual electricity use (kWh/year)
- Maintenance budget (€)

### 5.3 Enter Fleet Volumetry

In **Fleet volumetry (by device type)**:
- Enter the number of devices for each type
- The model automatically uses the volume of the currently selected type

### 5.4 Adjust France / LVMH Assumptions

You can modify:
- Electricity price (€/kWh)
- Grid carbon factor (kgCO₂e/kWh)

Defaults reflect the French electricity mix.

---

## 6. Understanding the Results

The app computes:
- Annualized cost per device (new vs refurbished)
- CO₂ footprint per device
- Fleet-level cost and CO₂ (for the selected type)
- Two comparable ROI scores:
  - ROI New
  - ROI Refurbished

The higher score indicates the better option.

All scores are capped between 0 and 100.

---

## 7. Scoring Logic (Simplified)

1. Maintenance increases device lifetime using a saturating function  
2. Refurbished lifetime is capped at 120% of new  
3. Cost and CO₂ are computed over the actual lifetime  
4. Scores are calculated as relative gains  
   - 50 = neutral  
   - >50 = better  
   - <50 = worse  
5. Global ROI score = 60% financial + 40% environmental  

---

## 8. Documentation & Sources

The app includes a **Documentation** section that lists:
- All parameters
- Their current values
- Units
- Justification
- Source or reference link

You can export this table as:
GreenIT_ROI_Documentation.csv


---

## 9. Exporting Results

You can export a full report:


It includes:
- Cost, CO₂, lifetime
- Fleet impact
- All key assumptions
- ROI scores (New & Refurbished)

---

## 10. Important Notes

- Brand modifiers, refurb ratios, and lifetime functions are explicit modeling assumptions  
- They should be calibrated with:
  - Supplier data
  - IT asset management data
  - LVMH procurement contracts  
- This is a scenario-based decision tool, not a regulatory carbon report
