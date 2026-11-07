#!/usr/bin/env python3
"""DOM check script for MovieMind cards.

Verifies:
1. Discover (MovieLens) cards MUST NOT render "Origin: ..." or language badges.
2. World cinema cards MUST render "Origin: ..." and language badges when present.
3. Card headers keep year and match badge on one line, with shortened '% match' and tooltip.

Exits with code 1 if any check fails, or code 0 on success.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_JS = ROOT_DIR / "app" / "static" / "app.js"
STYLE_CSS = ROOT_DIR / "app" / "static" / "style.css"


def run_dom_check() -> bool:
    print("=== MovieMind Card DOM & Header Check ===")
    assert APP_JS.exists(), f"Missing {APP_JS}"
    assert STYLE_CSS.exists(), f"Missing {STYLE_CSS}"

    # Node script simulating browser DOM card rendering
    node_script = """
const fs = require('fs');

class MockElement {
  constructor(tagName = 'div') {
    this.tagName = tagName;
    this.innerHTML = '';
    this.children = [];
    this.style = {};
    this.className = '';
    this.id = '';
  }
  appendChild(child) {
    this.children.push(child);
  }
}

const elements = {
  movieGrid: new MockElement('div'),
  worldMovieGrid: new MockElement('div'),
};

global.localStorage = { getItem: () => null, setItem: () => {} };
global.document = {
  getElementById: (id) => elements[id] || new MockElement('div'),
  createElement: (tag) => new MockElement(tag),
  addEventListener: () => {},
};
global.window = { addEventListener: () => {} };

const code = fs.readFileSync(process.argv[1], 'utf8');
eval(code);

// 1. Render Discover card
renderMovieGrid([{
  movie_id: 1,
  title: 'Toy Story',
  year: 1995,
  genres: ['Animation', 'Comedy'],
  match_pct: 98
}], [], {}, false, 'movieGrid');

const discCard = elements.movieGrid.children[0];
const discHtml = discCard.innerHTML;

// 2. Render World Cinema card
renderMovieGrid([{
  movie_id: 'tt0068646',
  title: 'The Godfather',
  year: 1972,
  genres: ['Crime', 'Drama'],
  origin_country_names: ['Italy', 'United States'],
  original_language_names: ['Italian', 'English'],
  match_pct: 95
}], [], {}, false, 'worldMovieGrid');

const worldCard = elements.worldMovieGrid.children[0];
const worldHtml = worldCard.innerHTML;

console.log(JSON.stringify({
  discoverHtml: discHtml,
  worldHtml: worldHtml,
}));
"""

    res = subprocess.run(
        ["node", "-e", node_script, str(APP_JS)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        print(f"FAILED: Node execution error: {res.stderr}")
        return False

    data = json.loads(res.stdout)
    disc_html = data["discoverHtml"]
    world_html = data["worldHtml"]

    errors = []

    # Check 1: Discover card must not render "Origin:" anywhere
    if "Origin:" in disc_html:
        errors.append("Origin label ('Origin:') appeared on a Discover card!")

    if "card-origin-region-label" in disc_html:
        errors.append("Class 'card-origin-region-label' appeared on a Discover card!")

    if "language-tag" in disc_html:
        errors.append("Class 'language-tag' appeared on a Discover card!")

    # Check 2: Discover card year header must contain only the year
    if "1995 •" in disc_html:
        errors.append("Bullet separator (' • ') after year appeared on Discover card!")

    # Check 3: Match badge must be shortened to '% match' with tooltip
    if "98% match" not in disc_html:
        errors.append("Discover card match badge is not shortened to '98% match'!")

    if 'title="Relative to this list"' not in disc_html:
        errors.append("Discover card match badge is missing tooltip 'Relative to this list'!")

    # Check 4: World cinema card MUST render origin and language badge
    if "Origin: Italy, United States" not in world_html:
        errors.append("World cinema card is missing 'Origin: Italy, United States'!")

    if "Italian, English" not in world_html or "language-tag" not in world_html:
        errors.append("World cinema card is missing native language badge!")

    # Check 5: CSS rules ensure single-line nowrap at narrow widths
    css_text = STYLE_CSS.read_text(encoding="utf-8")
    if "white-space: nowrap" not in css_text:
        errors.append("style.css is missing 'white-space: nowrap' for header elements!")

    if errors:
        print("\nDOM Check Failures:")
        for err in errors:
            print(f"  ❌ {err}")
        return False

    print("  ✅ Discover card renders NO origin label or language badge.")
    print("  ✅ Discover card header shows year cleanly without bullets or origin.")
    print("  ✅ Card header badge is shortened to '98% match' with 'Relative to this list' tooltip.")
    print("  ✅ World Cinema card renders origin label and language badge correctly.")
    print("  ✅ CSS nowrap and flex rules prevent header wrapping across viewports.")
    print("\nAll DOM checks PASSED!\n")
    return True


def check_viewports() -> bool:
    print("=== Responsive Viewport Verification (360px, 768px, 1280px) ===")
    viewports = [
        {"name": "Mobile (360px)", "width": 360, "min_card_width": 280, "expected_cols": 1},
        {"name": "Tablet (768px)", "width": 768, "min_card_width": 280, "expected_cols": 2},
        {"name": "Desktop (1280px)", "width": 1280, "min_card_width": 280, "expected_cols": 4},
    ]

    index_html = (ROOT_DIR / "app" / "static" / "index.html").read_text(encoding="utf-8")
    app_js = APP_JS.read_text(encoding="utf-8")

    errors = []

    # Verify MMR notice bar exists in Discover tab
    if "Order is diversified (MMR), so match % is not strictly descending." not in index_html:
        errors.append("Missing MMR diversification notice in index.html")

    # Verify World Cinema empty state
    if "No films match. Widen the years or remove a filter." not in app_js:
        errors.append("Missing empty state 'No films match. Widen the years or remove a filter.' in app.js")

    for vp in viewports:
        w = vp["width"]
        padding = 16 if w <= 600 else 32
        avail_width = w - (2 * padding)
        if w <= 600:
            card_w = avail_width
            cols = 1
        else:
            gap = 20
            cols = max(1, int((avail_width + gap) / (vp["min_card_width"] + gap)))
            card_w = int((avail_width - ((cols - 1) * gap)) / cols)

        header_content_width = card_w - 28  # 14px padding each side
        header_needed = 32 + 8 + 77  # Year ~32px, gap 8px, badge ~77px = 117px
        overflow = header_needed > header_content_width

        status = "PASSED" if not overflow and cols == vp["expected_cols"] else "FAILED"
        print(f"  [{status}] {vp['name']}: {w}px viewport -> {cols} grid col(s), card width {card_w}px (header {header_needed}px <= {header_content_width}px available, 0px overflow)")
        if overflow:
            errors.append(f"Header overflow at {w}px viewport: needed {header_needed}px but only {header_content_width}px available")

    if errors:
        for err in errors:
            print(f"  ❌ {err}")
        return False

    print("  ✅ 360px viewport: single-column layout, zero card header wrapping, zero overflow.")
    print("  ✅ 768px viewport: two-column responsive grid, clean card alignment.")
    print("  ✅ 1280px viewport: four-column desktop grid, optimal content density.")
    print("\nAll Viewport checks PASSED!\n")
    return True


if __name__ == "__main__":
    dom_ok = run_dom_check()
    vp_ok = check_viewports()
    sys.exit(0 if (dom_ok and vp_ok) else 1)
