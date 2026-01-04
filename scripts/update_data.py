
import pandas as pd
import numpy as np
import os

# Configuration
INPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'camera_data_clean4.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'camera_data_clean4.csv')

# Recommendation Presets (Keep these!)
# Extracted from main_agent.py
PRESET_KEYWORDS = [
    "ZV-E10", "G7 X", "Pocket", "Action", "Z30", # Vlog
    "X100", "GR III", "a6400", "Z fc", "X-T30", "X-S10", # Travel
    "Leica", "Pen-F", "X-E4", # Street
    "A7", "R6", "R5", "Z6", "Z5", "5D", # Portrait
    "Z7", "D850", "GFX", # Landscape
    "R50", "M50", "200D", "D3500", "a6000" # Beginner
]

# New Data to Append
# Filling as many columns as possible based on "camera_data_clean4.csv" schema
NEW_MODELS = [
    {
        "Brand": "DJI", "Model": "Osmo Pocket 3", "Year": 2023, 
        "Price": 3499, "Total megapixels": 9.4, "Effective megapixels": 9.4,
        "Sensor type": "CMOS", "Sensor size": '1" (13.2 x 8.8 mm)',
        "Weight_g": 179, "Dimensions": "139.7 x 42.2 x 33.5 mm",
        "Max_ISO": 6400, "Supports_4K": True, "Max. video resolution": "3840x2160 (120p)",
        "Screen_Size_in": 2.0, "Screen_Res_Dots": 314000, "Touchscreen": "Yes",
        "Storage types": "microSD", "Battery": "LiPo 1300 mAh",
        "Portability_Score": 98, "LowLight_Score": 70, "Video_Score": 95,
        "Also known as": "Pocket 3", "image_file": "data/images/dji_pocket_3.jpg"
    },
    {
        "Brand": "DJI", "Model": "Osmo Action 4", "Year": 2023,
        "Price": 2199, "Total megapixels": 10, "Effective megapixels": 10,
        "Sensor type": "CMOS", "Sensor size": '1/1.3"',
        "Weight_g": 145, "Dimensions": "70.5 x 44.2 x 32.8 mm",
        "Max_ISO": 12800, "Supports_4K": True, "Max. video resolution": "3840x2160 (120p)",
        "Screen_Size_in": 2.25, "Screen_Res_Dots": 360000, "Touchscreen": "Yes",
        "Storage types": "microSD", "Battery": "LiPo 1770 mAh",
        "Portability_Score": 99, "LowLight_Score": 65, "Video_Score": 92,
        "Also known as": "Action 4", "image_file": "data/images/dji_action_4.jpg"
    },
    {
        "Brand": "Olympus", "Model": "PEN-F", "Year": 2016,
        "Price": 5500, "Total megapixels": 21.8, "Effective megapixels": 20.3,
        "Sensor type": "Live MOS", "Sensor size": "Four Thirds (17.4 x 13 mm)",
        "Weight_g": 427, "Dimensions": "125 x 72 x 37 mm",
        "Max_ISO": 25600, "Supports_4K": False, "Max. video resolution": "1920x1080 (60p)",
        "Screen_Size_in": 3.0, "Screen_Res_Dots": 1037000, "Touchscreen": "Yes",
        "Storage types": "SD/SDHC/SDXC", "Battery": "BLN-1 Lithium-ion battery",
        "Portability_Score": 85, "LowLight_Score": 60, "Video_Score": 40,
        "Also known as": "Pen F", "image_file": "data/images/olympus_pen_f.jpg"
    }, # Kept despite age because it's in PROMPTS (Street)
    {
        "Brand": "Canon", "Model": "EOS 5D Mark IV", "Year": 2016,
        "Price": 11000, "Total megapixels": 31.7, "Effective megapixels": 30.4,
        "Sensor type": "CMOS", "Sensor size": "Full frame (36 x 24 mm)",
        "Weight_g": 890, "Dimensions": "151 x 116 x 76 mm",
        "Max_ISO": 32000, "Supports_4K": True, "Max. video resolution": "4096x2160 (30p)",
        "Screen_Size_in": 3.2, "Screen_Res_Dots": 1620000, "Touchscreen": "Yes",
        "Storage types": "CompactFlash + SD/SDHC/SDXC", "Battery": "LP-E6N lithium-ion battery",
        "Portability_Score": 30, "LowLight_Score": 88, "Video_Score": 75,
        "Also known as": "5D4", "image_file": "data/images/canon_5d_mark_iv.jpg"
    },
    {
        "Brand": "Fujifilm", "Model": "GFX 100S II", "Year": 2024,
        "Price": 35500, "Total megapixels": 102, "Effective megapixels": 102,
        "Sensor type": "BSI-CMOS", "Sensor size": "Medium format (44 x 33 mm)",
        "Weight_g": 883, "Dimensions": "150 x 104 x 87 mm",
        "Max_ISO": 12800, "Supports_4K": True, "Max. video resolution": "4096x2160 (30p)",
        "Screen_Size_in": 3.2, "Screen_Res_Dots": 2360000, "Touchscreen": "Yes",
        "Storage types": "SD/SDHC/SDXC (UHS-II)", "Battery": "NP-W235 lithium-ion battery",
        "Portability_Score": 40, "LowLight_Score": 98, "Video_Score": 80,
        "Also known as": "", "image_file": "data/images/fujifilm_gfx_100s_ii.jpg"
    },
    {
        "Brand": "Sony", "Model": "Alpha a6000", "Year": 2014,
        "Price": 3200, "Total megapixels": 24.7, "Effective megapixels": 24.3,
        "Sensor type": "CMOS", "Sensor size": "APS-C (23.5 x 15.6 mm)",
        "Weight_g": 344, "Dimensions": "120 x 67 x 45 mm",
        "Max_ISO": 25600, "Supports_4K": False, "Max. video resolution": "1920x1080 (60p)",
        "Screen_Size_in": 3.0, "Screen_Res_Dots": 921600, "Touchscreen": "No",
        "Storage types": "SD/SDHC/SDXC, Memory Stick Pro Duo", "Battery": "NP-FW50 lithium-ion battery",
        "Portability_Score": 88, "LowLight_Score": 65, "Video_Score": 50,
        "Also known as": "\u03b16000", "image_file": "data/images/sony_a6000.jpg"
    }
]

# Price Map for Existing Models (Approximate CNY)
PRICE_MAP = {
    "ZV-E10": 4500, "G7 X": 4200, "Z30": 4800,
    "X100V": 11000, "X100VI": 13000, "GR III": 7500, "a6400": 6000, "Z fc": 6500, "X-T30": 6800, "X-S10": 7500,
    "X-E4": 6500,
    "A7 III": 11000, "A7 IV": 15500, "A7C": 10500, "A7C II": 13500,
    "R6": 13000, "R6 Mark II": 15000, "R5": 21000, "R5 Mark II": 26000, "EOS R": 9000,
    "Z6": 8500, "Z6 II": 11000, "Z5": 7000, "Z7": 14000, "Z7 II": 18000, "D850": 17000,
    "R50": 4500, "M50": 4000, "M50 II": 4500, "200D": 3800, "250D": 4200, "D3500": 3500,
    "Z8": 24000, "Z9": 35000, "A1": 38000, "R3": 34000,
    "Leica Q3": 48000, "Leica Q2": 38000, "Leica Q": 20000, "Q3": 48000, "Q2": 38000
}

def is_kept(row):
    model = str(row['Model'])
    year = int(row['Year']) if pd.notnull(row['Year']) else 2000
    
    # Keep if new enough
    if year >= 2020:
        return True
    
    # Keep if likely in presets (fuzzy match)
    model_lower = model.lower()
    for kw in PRESET_KEYWORDS:
        if kw.lower() in model_lower:
            return True
            
    return False

def get_price(row):
    # Check manual map first
    model = str(row['Model'])
    for k, v in PRICE_MAP.items():
        if k in model:
            return v
    
    # Fallback simulation logic (similar to old loader but deterministic-ish)
    # Estimate based on sensor/year/score if needed, or just random range but smarter
    # Here we stick to a simplified range for unknowns
    return 8000 # Default fallback for unknowns

def update_cameras():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Add Price Column if missing
    if 'Price' not in df.columns:
        df['Price'] = np.nan
        
    # 2. Add 'Also known as' if missing (it seems present based on view_file, but ensure)
    if 'Also known as' not in df.columns:
        df['Also known as'] = ""

    # 3. Filter Old Models
    initial_count = len(df)
    df_kept = df[df.apply(is_kept, axis=1)].copy()
    print(f"Filtered {initial_count} -> {len(df_kept)} rows (removed outdated).")

    # 4. Append New Models
    new_rows = pd.DataFrame(NEW_MODELS)
    # Align columns
    for col in df_kept.columns:
        if col not in new_rows.columns:
            new_rows[col] = None 
            
    df_final = pd.concat([df_kept, new_rows], ignore_index=True)
    print(f"Added {len(new_rows)} new models. Total: {len(df_final)}")
    
    # 5. Populate Prices
    # Start with existing values, fill NaNs with our logic
    df_final['Price'] = df_final.apply(
        lambda r: r['Price'] if pd.notnull(r['Price']) and r['Price'] > 0 else get_price(r), 
        axis=1
    )
    
    # 6. Save
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    update_cameras()
