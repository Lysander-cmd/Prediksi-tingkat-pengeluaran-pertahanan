import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(csv_path="data/military_expenditure.csv", country=None, test_size=0.2, random_state=42):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Rename columns to simpler names
    df.columns = [col.strip() for col in df.columns]
    
    # Filter by country if specified
    if country:
        df = df[df['country'] == country]
        
    if df.empty:
        raise ValueError(f"No data found for country: {country}")
    
    # Clean and prepare data
    df = df.sort_values(by='year')
    
    # Remove rows with missing expenditure data
    expenditure_col = 'Military expenditure (current USD)'
    df = df.dropna(subset=[expenditure_col])
    
    # Convert year to numeric if it's not already
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # Prepare features and target
    X = df[['year']].copy()
    y = df[expenditure_col].values
    
    # Create additional features
    X['year_squared'] = X['year'] ** 2
    
    # Add GDP percentage feature if available
    gdp_col = 'Military expenditure (% of GDP)'
    if gdp_col in df.columns:
        X['gdp_pct'] = df[gdp_col].fillna(df[gdp_col].mean())
    
    # Add government expenditure percentage feature if available
    gov_exp_col = 'Military expenditure (% of general government expenditure)'
    if gov_exp_col in df.columns:
        X['gov_exp_pct'] = df[gov_exp_col].fillna(df[gov_exp_col].mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features (except year for reference)
    year_train = X_train['year'].copy()
    year_test = X_test['year'].copy()
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Add back the year column for reference (unscaled)
    X_train_scaled['year_original'] = year_train.values
    X_test_scaled['year_original'] = year_test.values
    
    return X_train_scaled, X_test_scaled, y_train, y_test