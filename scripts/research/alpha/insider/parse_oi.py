"""Parse OpenInsider 'tinytable' result tables (untrusted HTML -> rows). Usage: python -I parse_oi.py FILE.html"""
import sys, json
from html.parser import HTMLParser

class P(HTMLParser):
    def __init__(self):
        super().__init__(); self.in_t = False; self.depth = 0; self.rows = []; self.row = None; self.cell = None; self.hdr = []
        self.in_th = False
    def handle_starttag(self, tag, a):
        a = dict(a)
        if tag == "table" and "tinytable" in (a.get("class") or ""):
            self.in_t = True
        if not self.in_t: return
        if tag == "tr": self.row = []
        if tag in ("td", "th"): self.cell = ""; self.in_th = tag == "th"
    def handle_endtag(self, tag):
        if not self.in_t: return
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join(self.cell.split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            if self.row and self.in_th and not self.hdr: self.hdr = self.row
            elif self.row: self.rows.append(self.row)
            self.row = None
        if tag == "table": self.in_t = False
    def handle_data(self, d):
        if self.cell is not None: self.cell += d

p = P(); p.feed(open(sys.argv[1], encoding="utf-8", errors="replace").read())
hdr = [h.replace("\xa0", " ").strip() for h in p.hdr]
print(json.dumps({"header": hdr, "rows": [dict(zip(hdr, r)) for r in p.rows]}))
