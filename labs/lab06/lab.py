# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def _price_to_float(s: str) -> float:
    s = s.replace("Â", "").replace(",", "").strip()
    cleaned = []
    dot_used = False
    for ch in s:
        if ch.isdigit():
            cleaned.append(ch)
        elif ch == "." and not dot_used:
            cleaned.append(".")
            dot_used = True
    try:
        return float("".join(cleaned)) if cleaned else float("inf")
    except ValueError:
        return float("inf")


def _rating_from_classes(classes) -> str:
    for c in classes or []:
        if c in {"One", "Two", "Three", "Four", "Five"}:
            return c
    return "Unknown"


def _to_bare_product_path(href: str) -> str:
    href = href.strip()
    while href.startswith("../"):
        href = href[3:]
    while href.startswith("/"):
        href = href[1:]
    if href.startswith("catalogue/"):
        href = href[len("catalogue/"):]
    return href


def extract_book_links(text):
    soup = bs4.BeautifulSoup(text, "lxml")
    out = []
    for li in soup.select("ol.row > li"):
        pod = li.select_one("article.product_pod")
        if not pod:
            continue
        rating_word = _rating_from_classes((pod.select_one("p.star-rating") or {}).get("class", []))
        rating_val = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}.get(rating_word, 0)
        price_tag = pod.select_one(".price_color")
        if not price_tag:
            continue
        price_val = _price_to_float(price_tag.get_text(strip=True))
        if rating_val >= 4 and price_val < 50:
            a = pod.select_one("h3 a")
            if not a:
                continue
            href = a.get("href", "")
            out.append(_to_bare_product_path(href))
    return out


def get_product_info(text, categories):
    soup = bs4.BeautifulSoup(text, "lxml")
    title = soup.select_one("div.product_main h1")
    title = title.get_text(strip=True) if title else ""
    rating = _rating_from_classes((soup.select_one("p.star-rating") or {}).get("class", []))
    crumb = soup.select("ul.breadcrumb li a")
    category = crumb[2].get_text(strip=True) if len(crumb) >= 3 else None
    if category is None:
        return None
    allowed = {c.lower().strip() for c in categories}
    if category.lower().strip() not in allowed:
        return None
    info = {}
    for row in soup.select("table.table.table-striped tr"):
        th = row.select_one("th")
        td = row.select_one("td")
        if th and td:
            info[th.get_text(strip=True)] = td.get_text(" ", strip=True)
    desc = ""
    anchor = soup.select_one("#product_description")
    if anchor:
        p = anchor.find_next("p")
        if p:
            desc = p.get_text(" ", strip=True)
    return {
        "UPC": info.get("UPC", ""),
        "Product Type": info.get("Product Type", ""),
        "Price (excl. tax)": info.get("Price (excl. tax)", ""),
        "Price (incl. tax)": info.get("Price (incl. tax)", ""),
        "Tax": info.get("Tax", ""),
        "Availability": info.get("Availability", ""),
        "Number of reviews": info.get("Number of reviews", ""),
        "Category": category,
        "Rating": rating,
        "Description": desc,
        "Title": title,
    }


def scrape_books(k, categories):
    rows = []
    session = requests.Session()
    list_url = "http://books.toscrape.com/catalogue/page-{}.html"
    site_root = "http://books.toscrape.com/catalogue/"
    for page in range(1, k + 1):
        r = session.get(list_url.format(page), timeout=20)
        r.raise_for_status()
        bare_paths = extract_book_links(r.text)
        for path in bare_paths:
            product_url = site_root + path
            pr = session.get(product_url, timeout=20)
            if pr.status_code != 200:
                continue
            row = get_product_info(pr.text, categories)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


API_KEY = 'I5jwsn8YbxgPkdAa02TcoBSqV6AvYEnA'

def stock_history(ticker, year, month):
    start_date = f'{year}-{month:02d}-01'
    end_of_month = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
    end_date = end_of_month.strftime('%Y-%m-%d')

    url = (f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}'
           f'?from={start_date}&to={end_date}&apikey={API_KEY}')

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

    if 'historical' in data and data['historical']:
        history_df = pd.DataFrame(data['historical'])
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date').reset_index(drop=True)
        return history_df
    else:
        return pd.DataFrame()


def stock_stats(history):
    if history.empty:
        return ('+0.00%', '0.00B')

    start_price = history.iloc[0]['open']
    end_price = history.iloc[-1]['close']

    if start_price > 0:
        percent_change = ((end_price - start_price) / start_price) * 100
    else:
        percent_change = 0.0

    formatted_percent_change = f'{percent_change:+.2f}%'

    avg_daily_price = (history['high'] + history['low']) / 2
    daily_transaction_value = avg_daily_price * history['volume']
    total_transaction_value = daily_transaction_value.sum()
    total_transaction_billions = total_transaction_value / 1_000_000_000
    formatted_volume = f'{total_transaction_billions:.2f}B'

    return (formatted_percent_change, formatted_volume)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------
def _fetch_item(item_id):
    try:
        url = f'https://hacker-news.firebaseio.com/v0/item/{item_id}.json'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def get_comments(storyid):
    story_item = _fetch_item(storyid)
    if not story_item or 'kids' not in story_item:
        return pd.DataFrame(columns=['id', 'by', 'text', 'parent', 'time'])

    stack = story_item['kids'][::-1]
    comments_list = []

    while stack:
        comment_id = stack.pop()
        comment_item = _fetch_item(comment_id)
        
        if comment_item and not comment_item.get('dead', False):
            comments_list.append({
                'id': comment_item.get('id'),
                'by': comment_item.get('by'),
                'text': comment_item.get('text'),
                'parent': comment_item.get('parent'),
                'time': comment_item.get('time')
            })

            child_kids = comment_item.get('kids', [])
            stack.extend(child_kids[::-1])

    df = pd.DataFrame(comments_list)
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
    return df[['id', 'by', 'text', 'parent', 'time']]
