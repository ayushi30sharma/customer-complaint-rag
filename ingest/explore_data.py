import pandas as pd
import os

# Define file paths
DATA_DIR = "data"
COMPLAINTS_FILE = os.path.join(DATA_DIR, "ecommerce_customer_complaint_records.csv")
RESOLUTIONS_FILE = os.path.join(DATA_DIR, "ecommerce_customer_support_resolution_notes.csv")
RELEASES_FILE = os.path.join(DATA_DIR, "ecommerce_product_release_notes.csv")

# Load datasets
try:
    complaints = pd.read_csv(COMPLAINTS_FILE)
    resolutions = pd.read_csv(RESOLUTIONS_FILE)
    releases = pd.read_csv(RELEASES_FILE)
    print("✅ All datasets loaded successfully!\n")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# Display Complaints Data
print("=" * 50)
print("COMPLAINTS DATA")
print("=" * 50)
print(complaints.head())
print("\nDataset Info:")
print(complaints.info())
print(f"\nTotal Complaints: {len(complaints)}\n")

# Display Resolution Notes Data
print("=" * 50)
print("RESOLUTION NOTES DATA")
print("=" * 50)
print(resolutions.head())
print("\nDataset Info:")
print(resolutions.info())
print(f"\nTotal Resolutions: {len(resolutions)}\n")

# Display Product Release Notes Data
print("=" * 50)
print("PRODUCT RELEASE NOTES DATA")
print("=" * 50)
print(releases.head())
print("\nDataset Info:")
print(releases.info())
print(f"\nTotal Release Notes: {len(releases)}\n")

# Check for missing values
print("=" * 50)
print("MISSING VALUES CHECK")
print("=" * 50)
print("Complaints missing values:")
print(complaints.isnull().sum())
print("\nResolutions missing values:")
print(resolutions.isnull().sum())
print("\nReleases missing values:")
print(releases.isnull().sum())