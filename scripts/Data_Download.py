import requests
import pandas as pd
import json

# Create data directory
import os

os.makedirs("data/openfema", exist_ok=True)


# 1. Download NFIP Policies (small sample)
def download_fema_policies(limit=500):
    """Download sample of FIMA NFIP Redacted Policies v2"""
    base_url = "https://www.fema.gov/api/open/v2/FimaNfipPolicies"

    # Parameters for API
    params = {
        "$top": limit,  # Get 500 records
        "$skip": 0,
        "$filter": "reportedZipCode ne null",  # Filter out null zips
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Save as JSON
    with open("data/openfema/policies_sample.json", "w") as f:
        json.dump(data, f, indent=2)

    # Also save as CSV for easier inspection
    df = pd.DataFrame(data["FimaNfipPolicies"])
    df.to_csv("data/openfema/policies_sample.csv", index=False)

    print(f"Downloaded {len(df)} policy records")
    print(f"Columns: {list(df.columns)}")
    return df


# 2. Download NFIP Claims (small sample)
def download_fema_claims(limit=500):
    """Download sample of FIMA NFIP Redacted Claims v2"""
    base_url = "https://www.fema.gov/api/open/v2/FimaNfipClaims"

    params = {
        "$top": limit,
        "$skip": 0,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Save as JSON
    with open("data/openfema/claims_sample.json", "w") as f:
        json.dump(data, f, indent=2)

    # Also save as CSV
    df = pd.DataFrame(data["FimaNfipClaims"])
    df.to_csv("data/openfema/claims_sample.csv", index=False)

    print(f"Downloaded {len(df)} claim records")
    print(f"Columns: {list(df.columns)}")
    return df


# Run the downloads
if __name__ == "__main__":
    policies_df = download_fema_policies(500)
    claims_df = download_fema_claims(500)
