"""
scripts/resolve_all_wiki.py
Test resolution rate for all 444 places against Wikipedia and Wikimedia Commons.
"""
import concurrent.futures
import json
import os
import re
import urllib.parse
import urllib.request
import pandas as pd

USER_AGENT = "TripSenseBot/3.0 (contact@tripsense.org; research and educational use)"
HEADERS = {"User-Agent": USER_AGENT}

SKIP_PATTERNS = [
    ".svg", ".djvu", ".pdf", ".tif", ".tiff", ".gif",
    "logo", "icon", "map", "flag", "symbol", "coat of arms",
    "stamp", "coin", "diagram", "chart", "locator", "plan",
    "stub", "book", "inscription", "sketch"
]

def is_valid_image(title: str, mime: str = "") -> bool:
    t = title.lower()
    if any(skip in t for skip in SKIP_PATTERNS):
        return False
    if mime and not mime.startswith("image/"):
        return False
    if any(ext in t for ext in [".jpg", ".jpeg", ".png", ".webp"]):
        return True
    return False

def get_commons_file_info(file_name: str, width: int = 800):
    if not file_name.startswith("File:"):
        file_name = "File:" + file_name
    api_url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&titles="
        f"{urllib.parse.quote(file_name)}&prop=imageinfo&iiprop=url|mime|size&iiurlwidth={width}&format=json"
    )
    req = urllib.request.Request(api_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            pages = data.get("query", {}).get("pages", {})
            for _, p in pages.items():
                info = p.get("imageinfo", [{}])[0]
                url = info.get("thumburl") or info.get("url")
                desc = info.get("descriptionurl", f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(file_name)}")
                if url and is_valid_image(file_name, info.get("mime", "")):
                    return url, file_name, desc
    except Exception:
        pass
    return None, None, None

def get_wikidata_image(qid: str, width: int = 800):
    if not qid or not str(qid).startswith("Q"):
        return None, None, None
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            entity = data.get("entities", {}).get(qid, {})
            claims = entity.get("claims", {})
            # P18: Image
            if "P18" in claims:
                img_name = claims["P18"][0]["mainsnak"]["datavalue"]["value"]
                return get_commons_file_info(img_name, width=width)
            # Check enwiki sitelink if no direct P18
            sitelinks = entity.get("sitelinks", {})
            if "enwiki" in sitelinks:
                enwiki_title = sitelinks["enwiki"]["title"]
                return get_wiki_page_image(enwiki_title, width=width)
    except Exception:
        pass
    return None, None, None

def get_wiki_page_image(title: str, width: int = 800):
    if not title:
        return None, None, None
    # 1. Try summary API
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            thumb = data.get("thumbnail", {}).get("source")
            desc_url = data.get("content_urls", {}).get("desktop", {}).get("page")
            d_title = data.get("title", title)
            if thumb and is_valid_image(thumb):
                scaled = re.sub(r"/\d+px-", f"/{width}px-", thumb)
                return scaled, f"Wikipedia: {d_title}", desc_url
    except Exception:
        pass

    # 2. Try pageimages & images from query API
    api_url = (
        f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=pageimages|images&pithumbsize={width}&format=json"
    )
    req = urllib.request.Request(api_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            pages = data.get("query", {}).get("pages", {})
            for _, p in pages.items():
                if "thumbnail" in p:
                    src = p["thumbnail"].get("source")
                    if src and is_valid_image(src):
                        return src, f"Wikipedia: {title}", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                # fallback to first valid image from the page
                for img in p.get("images", []):
                    ititle = img.get("title", "")
                    if is_valid_image(ititle):
                        u, t, d = get_commons_file_info(ititle, width=width)
                        if u:
                            return u, t, d
    except Exception:
        pass

    return None, None, None

def search_commons(query: str, width: int = 800):
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit=5&prop=imageinfo"
        f"&iiprop=url|mime|size&iiurlwidth={width}&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            pages = data.get("query", {}).get("pages", {})
            for _, p in pages.items():
                title = p.get("title", "")
                info = p.get("imageinfo", [{}])[0]
                mime = info.get("mime", "")
                if is_valid_image(title, mime):
                    url = info.get("thumburl") or info.get("url")
                    desc = info.get("descriptionurl", f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}")
                    return url, title, desc
    except Exception:
        pass
    return None, None, None

def search_wikipedia_article(query: str):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&srlimit=3&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            results = data.get("query", {}).get("search", [])
            for res in results:
                t = res.get("title", "")
                snip = res.get("snippet", "").lower()
                # filter for relevance to Andhra / India / place
                if any(k in snip for k in ["andhra", "india", "temple", "district", "waterfall", "beach", "fort", "hills", "ghats", "river", "cave"]):
                    return t
            if results:
                return results[0].get("title", "")
    except Exception:
        pass
    return None

def resolve_place(row):
    pid = row["place_id"]
    name = row["name"]
    dist = str(row.get("district", "")).strip()
    cat = str(row.get("category", "")).strip()
    w_url = row.get("wiki_url")
    qid = row.get("wikidata_id")

    # Step 1: Try wikidata_id
    if pd.notna(qid):
        url, title, desc = get_wikidata_image(str(qid).strip())
        if url:
            return pid, url, title, desc, "wikidata"

    # Step 2: Try wiki_url
    if pd.notna(w_url) and "/wiki/" in str(w_url):
        w_title = urllib.parse.unquote(str(w_url).split("/wiki/")[1].split("#")[0].strip())
        url, title, desc = get_wiki_page_image(w_title)
        if url:
            return pid, url, title, desc, "wiki_url"
        # Also try searching commons with w_title
        url, title, desc = search_commons(w_title.replace("_", " "))
        if url:
            return pid, url, title, desc, "wiki_url_commons"

    # Clean name
    clean_name = re.sub(r"\s+(view\s*point|viewpoint|boating|falls|waterfall|temple|beach|fort)\b", "", name, flags=re.I).strip()

    # Step 3: Search Commons with specific queries
    queries = [
        f"{name} {dist}",
        f"{name} Andhra Pradesh",
        f"{clean_name} {dist}",
        f"{name}"
    ]
    for q in queries:
        url, title, desc = search_commons(q)
        if url:
            return pid, url, title, desc, f"commons_search:{q}"

    # Step 4: Search Wikipedia articles
    for q in [f"{name} {dist}", f"{name} Andhra Pradesh", f"{clean_name} {dist}"]:
        art_title = search_wikipedia_article(q)
        if art_title:
            url, title, desc = get_wiki_page_image(art_title)
            if url:
                return pid, url, title, desc, f"wikipedia_article:{art_title}"

    return pid, None, None, None, "unresolved"

if __name__ == "__main__":
    df = pd.read_csv("data/places/tripsense_places_v2.csv")
    print(f"Loaded {len(df)} places. Testing resolution with 16 workers...")
    
    rows = [r for _, r in df.iterrows()]
    resolved_count = 0
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for res in executor.map(resolve_place, rows):
            pid, url, title, desc, method = res
            results[pid] = {"url": url, "title": title, "desc": desc, "method": method}
            if url:
                resolved_count += 1
                
    print(f"Resolution complete: {resolved_count} / {len(df)} directly resolved!")
    with open("data/resolved_wiki_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
