import os
import sys
import json
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow importing config from project root
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_SCRIPTS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

# Ensure output directory exists
os.makedirs(config.OPENFEMA_DIR, exist_ok=True)


# 1. Download NFIP Policies (small sample)
def download_fema_policies(limit=500):
    """Download sample of FIMA NFIP Redacted Policies v2"""
    base_url = "https://www.fema.gov/api/open/v2/FimaNfipPolicies"

    params = {
        "$top": limit,
        "$skip": 0,
        "$filter": "reportedZipCode ne null",
    }

    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"ERROR: Failed to download FEMA policies — {exc}")
        sys.exit(1)

    out_json = os.path.join(config.OPENFEMA_DIR, "policies_sample.json")
    out_csv  = os.path.join(config.OPENFEMA_DIR, "policies_sample.csv")

    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    df = pd.DataFrame(data["FimaNfipPolicies"])
    df.to_csv(out_csv, index=False)

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

    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"ERROR: Failed to download FEMA claims — {exc}")
        sys.exit(1)

    out_json = os.path.join(config.OPENFEMA_DIR, "claims_sample.json")
    out_csv  = os.path.join(config.OPENFEMA_DIR, "claims_sample.csv")

    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    df = pd.DataFrame(data["FimaNfipClaims"])
    df.to_csv(out_csv, index=False)

    print(f"Downloaded {len(df)} claim records")
    print(f"Columns: {list(df.columns)}")
    return df


# Run the downloads
if __name__ == "__main__":
    policies_df = download_fema_policies(500)
    claims_df   = download_fema_claims(500)
