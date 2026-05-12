"""
Fetches article content from a URL or reads from a local file.
Strips HTML, extracts body text and basic metadata.
"""

import re
from pathlib import Path


def fetch_url(url: str) -> tuple[str, dict]:
    """Fetch a URL. Returns (body_text, metadata_dict). Requires: pip install requests beautifulsoup4"""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("URL fetching requires: pip install requests beautifulsoup4")

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    meta = {}
    title_tag = soup.find("title")
    meta["title"] = title_tag.get_text(strip=True) if title_tag else ""
    og_title = soup.find("meta", property="og:title")
    if og_title:
        meta["title"] = og_title.get("content", meta["title"])

    pub_date = soup.find("meta", property="article:published_time")
    meta["date"] = pub_date.get("content", "")[:10] if pub_date else ""

    author = soup.find("meta", attrs={"name": "author"})
    meta["journalist"] = author.get("content", "") if author else ""

    og_site = soup.find("meta", property="og:site_name")
    meta["org"] = og_site.get("content", "") if og_site else ""

    for tag in soup(["script", "style", "nav", "footer", "aside", "header",
                     "form", "button", "noscript", "figure", "figcaption"]):
        tag.decompose()

    body_text = ""
    for selector in ["article", "[class*='article-body']", "[class*='story-body']",
                     "[class*='post-body']", "[class*='content-body']", "main"]:
        candidate = soup.select_one(selector)
        if candidate:
            body_text = candidate.get_text(separator="\n", strip=True)
            break

    if not body_text:
        body = soup.find("body")
        body_text = body.get_text(separator="\n", strip=True) if body else ""

    return _clean_text(body_text), meta


def read_file(path: str) -> str:
    return _clean_text(Path(path).read_text(encoding="utf-8"))


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    # Use word count threshold — keeps short but valid sentences like "Police filed an FIR."
    lines = [line for line in lines if len(line.split()) >= 3 or line == ""]
    return "\n".join(lines).strip()
