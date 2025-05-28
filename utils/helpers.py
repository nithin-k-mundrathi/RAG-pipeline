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
    # usually have text, status_code, headers,json,cookies,url,content
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('div', {'class': 'mw-parser-output'})
    if not content:
        print(f"No main content found at {url}")
        return None

    # Remove infoboxes and tables (optional)
    # Decompose removes the table elements from the Content.
    for table in content.find_all('table'):
        table.decompose()

    # Remove unwanted sections
    unwanted_ids = ['References', 'Bibliography', 'External_links', 'See_also']
    # h2 and h3 are the headers here.
    for header in content.find_all(['h2', 'h3']):
        # Span is a content inside header.
        span = header.find('span', class_='mw-headline')
        # Spance can be the above references, bibliography or anything above.
        if span and span.get('id') in unwanted_ids:
            for sib in header.find_next_siblings():
                sib.decompose()
            header.decompose()
            break

    text = content.get_text(separator=' ', strip=True)
    return clean_text(text)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean_text_nltk(text):
    output_text = []
    sent = decontracted(text)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(e for e in sent.split())
    output_text.append(sent.lower().strip())
    return output_text[0]
