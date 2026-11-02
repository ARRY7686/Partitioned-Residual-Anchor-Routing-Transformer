"""Checks every refs.bib entry against bibliographic databases: DBLP by title, the arXiv API for
arXiv-only entries, and Semantic Scholar as a fallback. Prints, per entry, the database record
next to the bib fields and flags differences in title, authors, year and pages.

    python verify_refs.py refs.bib > refs_check.txt
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

UA = {"User-Agent": "refs-check/1.0 (academic citation verification)"}


def fetch(url):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read().decode("utf-8")


def plain(text):
    text = re.sub(r"\\[a-zA-Z]+|[{}\\'\"`^~]", "", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def similar(a, b):
    return SequenceMatcher(None, " ".join(plain(a)), " ".join(plain(b))).ratio()


def surnames(names):
    return [plain(n.split(",")[0] if "," in n else n.split()[-1])[-1] for n in names if plain(n)]


def parse_bib(text):
    entries = []
    for match in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", match.group(3) + "\n", re.S))
        fields["type"], fields["key"] = match.group(1), match.group(2)
        entries.append(fields)
    return entries


def dblp(title):
    data = json.loads(fetch("https://dblp.org/search/publ/api?format=json&h=8&q=" + urllib.parse.quote(" ".join(plain(title)))))
    hits = (data.get("result", {}).get("hits", {}) or {}).get("hit", []) or []
    records = []
    for hit in hits:
        info = hit["info"]
        authors = info.get("authors", {}).get("author", [])
        authors = [authors] if isinstance(authors, dict) else authors
        records.append({"title": info.get("title", ""), "authors": [re.sub(r" \d{4}$", "", a["text"]) for a in authors],
                        "year": info.get("year", ""), "venue": info.get("venue", ""), "pages": info.get("pages", ""),
                        "source": "DBLP " + info.get("type", "")})
    return records


def arxiv_entries(xml):
    records = []
    for entry in xml.split("<entry>")[1:]:
        title = re.search(r"<title>(.*?)</title>", entry, re.S)
        year = re.search(r"<published>(\d{4})", entry)
        ident = re.search(r"<id>http://arxiv.org/abs/([^<v]+)", entry)
        records.append({"title": re.sub(r"\s+", " ", title.group(1)) if title else "",
                        "authors": re.findall(r"<name>(.*?)</name>", entry),
                        "year": year.group(1) if year else "", "venue": "arXiv " + (ident.group(1) if ident else ""),
                        "pages": "", "source": "arXiv"})
    return records


def arxiv(eprint):
    return arxiv_entries(fetch("https://export.arxiv.org/api/query?id_list=" + eprint))


def openalex(title):
    data = json.loads(fetch("https://api.openalex.org/works?per-page=5&search=" + urllib.parse.quote(" ".join(plain(title)))))
    records = []
    for item in data.get("results", []):
        source = ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        biblio = item.get("biblio") or {}
        pages = f"{biblio.get('first_page')}-{biblio.get('last_page')}" if biblio.get("first_page") and biblio.get("last_page") else ""
        records.append({"title": item.get("title") or "", "year": str(item.get("publication_year", "")),
                        "authors": [a["author"]["display_name"] for a in item.get("authorships", [])],
                        "venue": source, "pages": pages, "source": "OpenAlex"})
    return records


def arxiv_search(title):
    words = [w for w in plain(title) if len(w) > 3][:6]
    query = "+AND+".join("ti:" + urllib.parse.quote(w) for w in words)
    return arxiv_entries(fetch("https://export.arxiv.org/api/query?max_results=5&search_query=" + query))


def crossref(title):
    data = json.loads(fetch("https://api.crossref.org/works?rows=5&query.bibliographic=" + urllib.parse.quote(title)))
    records = []
    for item in data["message"]["items"]:
        date = (item.get("published-print") or item.get("published-online") or item.get("issued") or {}).get("date-parts", [[""]])
        records.append({"title": (item.get("title") or [""])[0],
                        "authors": [f"{a.get('family', '')}, {a.get('given', '')}" for a in item.get("author", [])],
                        "year": str(date[0][0]) if date and date[0] else "",
                        "venue": (item.get("container-title") or [""])[0], "pages": item.get("page", ""),
                        "source": "Crossref " + item.get("DOI", "")})
    return records


def semantic_scholar(title):
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?limit=5&fields=title,authors,year,venue&query="
           + urllib.parse.quote(title))
    data = json.loads(fetch(url))
    return [{"title": p.get("title", ""), "authors": [a["name"] for a in p.get("authors", [])], "year": str(p.get("year", "")),
             "venue": p.get("venue", ""), "pages": "", "source": "Semantic Scholar"} for p in data.get("data", [])]


def best(entry, records):
    scored = sorted(records, key=lambda r: -similar(entry["title"], r["title"]))
    return scored[0] if scored else None


def main(path):
    entries = parse_bib(open(path, encoding="utf-8").read())
    print(f"{len(entries)} entries\n")
    for entry in entries:
        records = []
        sources = [lambda: arxiv(entry["eprint"])] if entry.get("eprint") else []
        sources += [lambda: crossref(entry["title"]), lambda: openalex(entry["title"]), lambda: arxiv_search(entry["title"])]
        for source in sources:
            try:
                records += source()
            except Exception as error:   # note and try the next database
                print(f"   ({entry['key']}: one lookup failed: {error})")
            time.sleep(3.0)
            if records and similar(entry["title"], best(entry, records)["title"]) >= 0.95:
                break
        record = best(entry, records)
        bib_authors = re.split(r"\s+and\s+", entry.get("author", ""))
        flags = []
        if record is None:
            print(f"!! {entry['key']}: NOT FOUND in any database\n")
            continue
        title_score = similar(entry["title"], record["title"])
        if title_score < 0.9:
            flags.append(f"title similarity {title_score:.2f}")
        bib_s, db_s = surnames(bib_authors), surnames(record["authors"])
        if bib_s != db_s:
            flags.append(f"authors differ: bib {bib_s} vs db {db_s}")
        if entry.get("year", "") != record["year"]:
            flags.append(f"year bib {entry.get('year')} vs db {record['year']}")
        if entry.get("pages") and record["pages"] and entry["pages"].replace("--", "-") != record["pages"]:
            flags.append(f"pages bib {entry['pages']} vs db {record['pages']}")
        status = "OK " if not flags else "?? "
        print(f"{status}{entry['key']}  [{record['source']}]  {record['venue']} {record['year']} pp.{record['pages']}")
        print(f"    db title: {record['title']}")
        for flag in flags:
            print(f"    - {flag}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
