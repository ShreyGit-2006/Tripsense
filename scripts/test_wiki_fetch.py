"""
test_wiki_fetch.py
"""
import urllib.request
import urllib.parse
import json
import re
import pandas as pd

USER_AGENT = "TripSenseBot/3.0 (contact@tripsense.org; research and educational use)"
HEADERS = {"User-Agent": USER_AGENT}

def get_commons_file_url(file_name, width=800):
    if not file_name.startswith("File:"):
        file_name = "File:" + file_name
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(file_name)}&prop=imageinfo&iiprop=url|mime&iiurlwidth={width}&format=json"
    req = urllib.request.Request(api_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            pages = data.get("query", {}).get("pages", {})
            for _, p in pages.items():
                info = p.get("imageinfo", [{}])[0]
                return info.get("thumburl") or info.get("url"), p.get("title"), info.get("descriptionurl")
    except Exception:
        pass
    return None, None, None

def get_wikidata_image(qid, width=800):
    if not qid or not str(qid).startswith("Q"):
        return None, None, None
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            entity = data.get("entities", {}).get(qid, {})
            claims = entity.get("claims", {})
            if "P18" in claims:
                img_name = claims["P18"][0]["mainsnak"]["datavalue"]["value"]
                return get_commons_file_url(img_name, width=width)
    except Exception:
        pass
    return None, None, None

def get_wiki_summary_image(title, width=800):
    if not title:
        return None, None, None
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            thumb = data.get("thumbnail", {}).get("source")
            desc_url = data.get("content_urls", {}).get("desktop", {}).get("page")
            t = data.get("title", title)
            if thumb:
                # scale to desired width
                scaled = re.sub(r"/\d+px-", f"/{width}px-", thumb)
                return scaled, f"Wikipedia: {t}", desc_url
    except Exception:
        pass
    return None, None, None

def search_commons(query, width=800):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit=5&prop=imageinfo&iiprop=url|mime&iiurlwidth={width}&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            pages = data.get("query", {}).get("pages", {})
            for _, p in pages.items():
                title = p.get("title", "")
                info = p.get("imageinfo", [{}])[0]
                mime = info.get("mime", "")
                # skip svg, logo, map, icon
                if "image" in mime and not any(skip in title.lower() for skip in [".svg", "logo", "icon", "map", "flag", "symbol", "coat of arms"]):
                    return info.get("thumburl") or info.get("url"), title, info.get("descriptionurl")
    except Exception:
        pass
    return None, None, None

def search_wikipedia_title(query):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&srlimit=3&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            results = data.get("query", {}).get("search", [])
            for res in results:
                title = res.get("title", "")
                snippet = res.get("snippet", "").lower()
                # prioritize if snippet mentions andhra or district or india or category
                return title
    except Exception:
        pass
    return None

if __name__ == "__main__":
    df = pd.read_csv("data/places/tripsense_places_v2.csv")
    print("Testing first 15 rows:")
    for _, r in df.head(15).iterrows():
        pid = r["place_id"]
        name = r["name"]
        dist = r["district"]
        w_url = r["wiki_url"]
        qid = r["wikidata_id"]
        
        url, title, desc = None, None, None
        
        # 1. Try wikidata
        if pd.notna(qid):
            url, title, desc = get_wikidata_image(str(qid).strip())
        
        # 2. Try wiki_url
        if not url and pd.notna(w_url) and "/wiki/" in str(w_url):
            w_title = urllib.parse.unquote(str(w_url).split("/wiki/")[1].split("#")[0].strip())
            url, title, desc = get_wiki_summary_image(w_title)
            if not url:
                url, title, desc = search_commons(w_title)
                
        # 3. Search commons with name + district
        if not url:
            url, title, desc = search_commons(f"{name} {dist}")
            
        # 4. Search commons with name
        if not url:
            url, title, desc = search_commons(name)
            
        # 5. Search wikipedia title
        if not url:
            wt = search_wikipedia_title(f"{name} {dist} Andhra Pradesh")
            if wt:
                url, title, desc = get_wiki_summary_image(wt)
                
        print(f"[{pid}] {name} ({dist}) => {title or 'NOT FOUND'} | {desc or ''}")
