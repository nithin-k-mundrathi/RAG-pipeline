import requests
from bs4 import BeautifulSoup
import re

def clean_text(content):
    return re.sub(r'\[\d+\]', '', content)

def fetch_and_clean(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('div', {'class': 'mw-parser-output'})
    if not content:
        print(f"No main content found at {url}")
        return None

    # Remove infoboxes and tables (optional)
    for table in content.find_all('table'):
        table.decompose()

    # Remove unwanted sections
    unwanted_ids = ['References', 'Bibliography', 'External_links', 'See_also']
    for header in content.find_all(['h2', 'h3']):
        span = header.find('span', class_='mw-headline')
        if span and span.get('id') in unwanted_ids:
            for sib in header.find_next_siblings():
                sib.decompose()
            header.decompose()
            break

    text = content.get_text(separator=' ', strip=True)
    return clean_text(text)
