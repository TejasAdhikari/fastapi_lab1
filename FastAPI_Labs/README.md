# Wine Classification API

## Overview
Machine Learning API to classify wine types using Random Forest Classifier.

## Dataset
Wine dataset with 13 chemical properties of wines from 3 different cultivars.

## Features
- alcohol: Alcohol content
- malic_acid: Malic acid content
- ash: Ash content
- alcalinity_of_ash: Alcalinity of ash
- magnesium: Magnesium content
- total_phenols: Total phenols
- flavanoids: Flavanoids content
- nonflavanoid_phenols: Nonflavanoid phenols
- proanthocyanins: Proanthocyanins content
- color_intensity: Color intensity
- hue: Hue
- od280_od315_of_diluted_wines: OD280/OD315 of diluted wines
- proline: Proline content

## Model
Random Forest Classifier with 100 estimators

## Modifications from Original
- Changed dataset from Iris to Wine
- Changed model from Decision Tree to Random Forest
- Updated API endpoints for wine features
- Achieved 98%+ test accuracy

## Setup
1. Create virtual environment: `python -m venv fastapi_lab1_env`
2. Activate: `source fastapi_lab1_env/bin/activate` (Mac/Linux) or `fastapi_lab1_env\Scripts\activate` (Windows)
3. Install: `pip install -r requirements.txt`
4. Train model: `cd src && python train.py`
5. Run server: `uvicorn main:app --reload`
6. Test: http://127.0.0.1:8000/docs

## API Endpoints
- GET `/` - Health check
- POST `/predict` - Predict wine class