#!/usr/bin/env python3
"""Zotero local API query tool. Usage: python zotero-query.py <command> [args]"""
import sys, json, requests, urllib.parse

BASE = "http://localhost:23119/api/users/0"

def api(path, params=None):
    r = requests.get(f"{BASE}/{path}", params=params or {})
    r.raise_for_status()
    return r.json()

def search(query, limit=20):
    """Search items by title/creator/tag keywords."""
    items = api("items", {"q": query, "limit": limit, "itemType": "-attachment"})
    results = []
    for it in items:
        d = it["data"]
        creators = ", ".join(c.get("lastName", c.get("name", "")) for c in d.get("creators", []))
        results.append({
            "key": d["key"], "type": d["itemType"], "title": d.get("title", ""),
            "creators": creators, "date": d.get("date", ""),
            "tags": [t["tag"] for t in d.get("tags", [])],
        })
    return results

def detail(key):
    """Get full metadata for an item."""
    it = api(f"items/{key}")
    d = it["data"]
    # get children (attachments)
    children = api(f"items/{key}/children")
    pdfs = []
    for c in children:
        if c["data"].get("contentType") == "application/pdf":
            link = c.get("links", {}).get("enclosure", {})
            path = urllib.parse.unquote(link.get("href", "").replace("file://", ""))
            pdfs.append({"key": c["data"]["key"], "path": path, "title": c["data"].get("title", "")})
    return {**d, "pdfs": pdfs}

def collections(parent=None):
    """List collections, optionally under a parent."""
    path = f"collections/{parent}/collections" if parent else "collections"
    return [{"key": c["data"]["key"], "name": c["data"]["name"],
             "numItems": c["meta"].get("numItems", 0)} for c in api(path)]

def collection_items(key, limit=50):
    """List items in a collection."""
    items = api(f"collections/{key}/items", {"limit": limit, "itemType": "-attachment"})
    return [{"key": it["data"]["key"], "title": it["data"].get("title", ""),
             "type": it["data"]["itemType"]} for it in items]

def fulltext(key):
    """Get full-text content of an item (indexed by Zotero)."""
    try:
        children = api(f"items/{key}/children")
        for c in children:
            if c["data"].get("contentType") == "application/pdf":
                ft = api(f"items/{c['data']['key']}/fulltext")
                return ft.get("content", "")
        return ""
    except: return ""

def tags(limit=100):
    """List all tags."""
    return [t["tag"] for t in api("tags", {"limit": limit})]

def pdf_paths(keys):
    """Get PDF file paths for given item keys."""
    paths = {}
    for key in keys:
        d = detail(key)
        for p in d.get("pdfs", []):
            if p["path"]:
                paths[key] = p["path"]
                break
    return paths

def library_version():
    """Get current library version for sync tracking."""
    r = requests.get(f"{BASE}/items", params={"limit": 1})
    return int(r.headers.get("Last-Modified-Version", 0))

def items_since(version, limit=100):
    """Get items modified since a library version."""
    items = api("items", {"since": version, "limit": limit, "itemType": "-attachment"})
    return [{"key": it["data"]["key"], "title": it["data"].get("title", ""),
             "version": it["version"]} for it in items]

def deleted_since(version):
    """Get items deleted since a library version."""
    r = requests.get(f"{BASE}/deleted", params={"since": version})
    r.raise_for_status()
    return r.json().get("items", [])

if __name__ == "__main__":
    cmds = {"search": search, "detail": detail, "collections": collections,
            "collection_items": collection_items, "fulltext": fulltext, "tags": tags,
            "pdf_paths": pdf_paths, "library_version": library_version,
            "items_since": items_since, "deleted_since": deleted_since}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"Usage: {sys.argv[0]} <{'|'.join(cmds)}> [args...]")
        sys.exit(1)
    cmd = cmds[sys.argv[1]]
    args = sys.argv[2:]
    # parse args: if single arg looks like list, split by comma
    if len(args) == 1 and "," in args[0]:
        args = [args[0].split(",")]
    result = cmd(*args) if args else cmd()
    print(json.dumps(result, ensure_ascii=False, indent=2))
