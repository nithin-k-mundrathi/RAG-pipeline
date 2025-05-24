import requests
from bs4 import BeautifulSoup
import re

def clean_text(content):
    # Remove references that usually appear as [1], [2], etc.
    content = re.sub(r'\[\d+\]', '', content)
    return content

def fetch_and_clean(url):
    # Fetch the content of the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the main content of the article, ignoring side boxes and headers
    content = soup.find('div', {'class': 'mw-parser-output'})
    # Remove the bibliography section, which generally follows a header like  "References", "Bibliography"
    for section_title in ['References', 'Bibliography', 'External links', 'See also']:
        section = content.find('span', id=section_title)
        if section:
        # Remove all content from this section to the end of the document
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()
        # Extract and clean the text
    text = content.get_text(separator=' ', strip=True)
    text = clean_text(text)
    return text