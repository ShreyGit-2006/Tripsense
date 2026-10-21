"""
scripts/replace_places_with_user_images.py
==========================================
Replaces place images in TripSense with authentic photographs from C:\\Users\\anish\\Desktop\\Tripsense\\Images.
Updates:
1. data/images/places/<place_id>.jpg
2. static/place_<place_id>.jpg
3. static/<place_id>.jpg
4. data/places/place_image_manifest.csv
5. data/images/places_metadata.json
"""

import csv
import json
import os
import re
import shutil
import pandas as pd
from PIL import Image
from difflib import SequenceMatcher

USER_IMAGES_DIR = r"C:\Users\anish\Desktop\Tripsense\Images"
PLACES_CSV = "data/places/tripsense_places_v2.csv"
DATA_IMG_DIR = "data/images/places"
STATIC_DIR = "static"
MANIFEST_PATH = "data/places/place_image_manifest.csv"
METADATA_PATH = "data/images/places_metadata.json"

os.makedirs(DATA_IMG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

df_places = pd.read_csv(PLACES_CSV)

def norm(text):
    text = str(text).lower()
    text = re.sub(r'[\(\)\[\],_\-\.\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

places = []
for _, r in df_places.iterrows():
    places.append({
        'id': r['place_id'],
        'name': r['name'],
        'norm': norm(r['name']),
        'district': str(r['district']),
        'norm_dist': norm(r['district']),
    })

manual_map = {
    "Adurru Buddhist.jpg": "AP0244",
    "Alluru Region (NTR).jpg": "AP0407",
    "Ancient Shiva Temples.jpg": "AP0311",
    "Anupu is a heritage in Palnadu.jpg": "AP0434",
    "Bhattiprolu railway station.jpg": "AP0433",
    "Chowdeswari Devi Temple.jpg": "AP0328",
    "City Central park.jpg": "AP0437",
    "Guntupalli Caves.jpg": "AP0442",
    "Indranatheswara Temple.jpg": "AP0366",
    "Kalingapatnam Lighthouse & Beach.jpg": "AP0423",
    "Kambadur Temple.jpg": "AP0376",
    "Mudasarlova Reservoir is a reservoir in Visakhapatnam.jpg": "AP0440",
    "Munagacharla Heritage Area.jpg": "AP0406",
    "Navanandi Temples.jpg": "AP0341",
    "Nawabs Palace.jpg": "AP0254",
    "Pushpagiri Temple Complex.jpg": "AP0367",
    "Sivajipalem is a  locality in Visakhapatnam.jpg": "AP0438",
    "Somanna Waterfalls (Palnadu)..jpg": "AP0009",
    "Talupulamma Temple.jpg": "AP0414",
    "Tripuranthakeswara Temple.jpg": "AP0388",
    "Udayagiri Fort Temple.jpg": "AP0359",
    "Yaganti Cave Temple.jpg": "AP0327",
    "Yarada is a neighborhood situated on the  Visakhapatnam City.jpg": "AP0426",
    "Venkateswara Swami Tirupati.jpg": "AP0345",
    "Nelepattu Bird Tirupati.webp": "AP0404",
    "Kilaru Anjaneya Swamy.jpg": "AP0242",
    "Ganga Parameswari Temple.jpg": "AP0352",
    "Kamalasaraswathi Temple.jpg": "AP0335",
    "Palliswaraswamy Temple (Kalakada).jpg": "AP0249",
    "Swayambhu Lingeswara Temple.jpg": "AP0287",
    "Peddamudium (YSR Kadapa).jpg": "AP0365",
    "Eluru.jpg": "AP0034",
}

place_by_id = {p['id']: p for p in places}

files = [f for f in os.listdir(USER_IMAGES_DIR) if not f.endswith('.csv')]

matches = {}
for f in sorted(files):
    if f in manual_map and manual_map[f] in place_by_id:
        p = place_by_id[manual_map[f]]
        matches[f] = (p['id'], p['name'])
        continue

    stem = os.path.splitext(f)[0]
    n_stem = norm(stem)
    
    # Exact
    exact = [p for p in places if n_stem == p['norm']]
    if exact:
        matches[f] = (exact[0]['id'], exact[0]['name'])
        continue
        
    # Containment
    best_p = None
    best_score = 0.0
    for p in places:
        if n_stem in p['norm'] or p['norm'] in n_stem:
            shorter = min(len(n_stem), len(p['norm']))
            longer = max(len(n_stem), len(p['norm']))
            score = 0.85 + 0.14 * (shorter / longer)
            if score > best_score:
                best_score = score
                best_p = p
                
    if best_score >= 0.85:
        matches[f] = (best_p['id'], best_p['name'])
        continue
        
    # Token overlap
    t_stem = set(n_stem.split()) - {'the', 'and', 'at', 'in', 'of', 'is', 'a', 'area', 'region', 'tourist', 'place'}
    for p in places:
        t_p = set(p['norm'].split()) - {'the', 'and', 'at', 'in', 'of', 'is', 'a', 'area', 'region', 'tourist', 'place'}
        overlap = t_stem & t_p
        if overlap:
            seq = SequenceMatcher(None, n_stem, p['norm']).ratio()
            jaccard = len(overlap) / len(t_stem | t_p)
            score = 0.5 * jaccard + 0.5 * seq
            if p['norm_dist'] in n_stem:
                score += 0.15
            if score > best_score:
                best_score = score
                best_p = p
                
    if best_score >= 0.50:
        matches[f] = (best_p['id'], best_p['name'])

print(f"Total images matched to destinations: {len(matches)}")

# Load existing metadata if present
metadata = {}
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as mf:
            metadata = json.load(mf)
    except Exception:
        metadata = {}

converted_count = 0
for filename, (pid, pname) in matches.items():
    src_path = os.path.join(USER_IMAGES_DIR, filename)
    dest_data_img = os.path.join(DATA_IMG_DIR, f"{pid}.jpg")
    dest_static_place = os.path.join(STATIC_DIR, f"place_{pid}.jpg")
    dest_static_direct = os.path.join(STATIC_DIR, f"{pid}.jpg")
    
    try:
        # Convert to RGB JPEG
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            # Save high-quality JPEG
            rgb_img.save(dest_data_img, "JPEG", quality=90, optimize=True)
            rgb_img.save(dest_static_place, "JPEG", quality=90, optimize=True)
            rgb_img.save(dest_static_direct, "JPEG", quality=90, optimize=True)
            
        metadata[pid] = {
            "place_name": pname,
            "filename": f"{pid}.jpg",
            "caption": f"{pname} (Authentic Photo)",
            "source": filename,
            "status": "user_provided_authentic"
        }
        converted_count += 1
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save updated metadata
with open(METADATA_PATH, "w", encoding="utf-8") as mf:
    json.dump(metadata, mf, indent=2, ensure_ascii=False)

# Update place_image_manifest.csv
if os.path.exists(MANIFEST_PATH):
    manifest_df = pd.read_csv(MANIFEST_PATH)
    pid_to_name = {m[0]: m[1] for m in matches.values()}
    
    for idx, row in manifest_df.iterrows():
        pid = row.get("place_id")
        if pid in pid_to_name:
            manifest_df.at[idx, "thumbnail_url"] = f"data/images/places/{pid}.jpg"
            manifest_df.at[idx, "image_title"] = f"{pid_to_name[pid]} (Authentic Photo)"
            manifest_df.at[idx, "verified"] = "True"
            manifest_df.at[idx, "match_status"] = "User Verified Authentic Image"
            
    manifest_df.to_csv(MANIFEST_PATH, index=False)

print(f"Successfully converted and replaced {converted_count} place images across data/ and static/!")
