import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

BASE = "https://www.transfermarkt.com"
LEAGUE_URL = "https://www.transfermarkt.com/eredivisie/startseite/wettbewerb/NL1/plus/?saison_id=2024"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8,nl;q=0.8,ko;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
}


def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retries = Retry(
        total=5, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def get_soup(session, url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def extract_club_squad_links(session) -> list[tuple[str, str]]:
    """
    Returns list of (club_name, squad_url)
    from the league table by reading the anchor on the 'Squad' column.
    """
    soup = get_soup(session, LEAGUE_URL)
    # Main club table usually has id='yw1' or class='items'
    table = soup.select_one("table.items") or soup.select_one("table#yw1")
    if not table:
        raise RuntimeError("League clubs table not found. Markup may have changed.")
    out = []
    for row in table.select("tbody > tr"):
        # Club name
        club_a = row.select_one("td:nth-of-type(2) a[href*='/verein/']")
        club_name = club_a.get_text(strip=True) if club_a else None

        # Link inside the Squad column (usually the 3rd TD)
        squad_a = row.select_one("td:nth-of-type(3) a[href]")
        if club_name and squad_a and squad_a.get("href"):
            squad_href = urljoin(BASE, squad_a["href"])
            out.append((club_name, squad_href))
    if not out:
        raise RuntimeError("No squad links found on the league page.")
    return out


def clean_text(x: str | None) -> str | None:
    if x is None:
        return None
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else None


def parse_player_rows_from_compact_table(soup: BeautifulSoup) -> list[dict]:
    """
    Robust parser for the 'Compact' squad table on Transfermarkt.
    """
    table = soup.select_one("table.items")
    if not table:
        return []

    players = []
    for tr in table.select("tbody > tr"):
        # Some rows are spacer rows (e.g. 'bg_blau_20'), skip them
        if tr.get("class") and any("bg_" in c for c in tr["class"]):
            continue

        # 1) Jersey number: text of the first 'zentriert' cell or the first td
        number = None
        td_num = tr.select_one("td.zentriert")
        if td_num:
            number = clean_text(td_num.get_text())
            # The jersey number cell might actually be nationality/age, so keep only numbers
            if number and not re.fullmatch(r"\d{1,3}", number):
                number = None
        # Fallback: the first td may contain the number
        if number is None:
            maybe = tr.select_one("td")
            if maybe:
                txt = clean_text(maybe.get_text())
                if txt and re.fullmatch(r"\d{1,3}", txt):
                    number = txt

        # 2) Name & position: usually inside the inline-table in the second td
        name, position = None, None
        name_a = tr.select_one("td:nth-of-type(2) a.spielprofil_tooltip, td.hauptlink a")
        if name_a:
            name = clean_text(name_a.get_text())

        # The position appears as the small text in the same cell (second row) or in a separate cell
        # Typical pattern: second tr > td inside the inline-table
        pos_cells = tr.select("td:nth-of-type(2) table.inline-table tr:nth-of-type(2) td")
        if pos_cells:
            position = clean_text(pos_cells[-1].get_text())
        if not position:
            # Alternative patterns
            pos_alt = tr.select_one("td:nth-of-type(2) .posrela, td:nth-of-type(2) .inline-table tr+tr td")
            if pos_alt:
                position = clean_text(pos_alt.get_text())

        # 3) Market value: usually the last column 'rechts hauptlink'
        mv_cell = tr.select_one("td.rechts.hauptlink, td.rechts.nowrap, td.rechts")
        market_value = None
        if mv_cell:
            mv_a = mv_cell.select_one("a")
            market_value = clean_text(mv_a.get_text() if mv_a else mv_cell.get_text())
            # Keep only formats like "€6.00m"
            mv_match = re.search(r"€[\d\.,]+\s*[a-zA-ZkKmMbB]*", market_value or "")
            if mv_match:
                market_value = mv_match.group(0)

        # Collect only valid player rows
        if name:
            players.append(
                {
                    "uniform_number": number,
                    "player_name": name,
                    "position": position,
                    "market_value": market_value,
                }
            )
    return players


def scrape_team(session, club_name: str, squad_url: str) -> pd.DataFrame:
    """
    The squad_url is the link attached to the 'Squad' column count.
    That page typically defaults to the 'Compact' tab, which exposes the same table.items structure.
    """
    soup = get_soup(session, squad_url)

    # If the page is not already on the 'Compact' tab, follow the compact tab link
    compact_tab = soup.select_one("a#statistik-tabs-0, a[href*='show=Kompakt'], a[href*='show=compact']")
    if compact_tab and compact_tab.get("href"):
        # May be a relative URL
        compact_url = urljoin(BASE, compact_tab["href"])
        soup = get_soup(session, compact_url)

    rows = parse_player_rows_from_compact_table(soup)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "team_name", club_name)
    return df


def main():
    session = make_session()
    teams = extract_club_squad_links(session)

    frames = []
    for i, (club, url) in enumerate(teams, start=1):
        print(f"[{i}/{len(teams)}] {club} -> {url}")
        try:
            df_team = scrape_team(session, club, url)
            if not df_team.empty:
                frames.append(df_team)
        except Exception as e:
            print(f"  ! Failed for {club}: {e}")
        time.sleep(1.2)  # Polite delay

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=["team_name", "uniform_number", "player_name", "position", "market_value"])
    all_df["uniform_number"] = all_df["uniform_number"].astype("string")
    all_df["player_name"] = all_df["player_name"].astype("string")
    all_df["position"] = all_df["position"].astype("string")
    all_df["market_value"] = all_df["market_value"].astype("string")

    out_path = "data/ajax/transfermarkt_2425.csv"
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(all_df):,} rows -> {out_path}")

    return all_df


if __name__ == "__main__":
    df = main()
    print(df.head(10).to_string(index=False))
