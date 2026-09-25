"""
Text cleaning and canonical normalization utilities for Business Entity Resolution.
Handles US, India, and France address/name noise patterns.
"""

import re
import unicodedata
from typing import Optional, Set, List


# Common business legal suffixes across US, India, and France
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'pvt ltd', 'co', 'company', 'group', 'holdings',
    'enterprises', 'associates', 'llp', 'pllc', 'gmbh', 'sarl', 'sas',
    'sa', 'eurl', 'sasu', 'sci', 'snc', 'gie', 'spa', 'bv'
}

# Domain extensions to strip when names are web URLs
DOMAIN_EXTENSIONS = re.compile(r'\.(com|in|org|net|co|io|fr|us|biz|info|ai|edu|gov)(/.*)?$', re.IGNORECASE)

# Standard address abbreviation mappings
ADDRESS_ABBR = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'av': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'sq': 'square',
    'hwy': 'highway',
    'pkwy': 'parkway',
    'ste': 'suite',
    'apt': 'apartment',
    'dept': 'department',
    'fl': 'floor',
    'bldg': 'building',
    'opp': 'opposite',
    'nr': 'near',
    'dist': 'district',
    'sec': 'sector',
    'pk': 'park',
}


def unicode_normalize(text: str) -> str:
    """Normalize unicode characters (e.g. accents in French addresses)."""
    if not text:
        return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')


def clean_name(name: Optional[str]) -> str:
    """
    Cleans and canonicalizes a business name:
    - Strips URL prefixes/suffixes (www., .com, etc.)
    - Removes punctuation and standardizes legal suffixes
    - Retains core alphanumeric tokens
    """
    if not name or str(name).lower() == 'nan':
        return ""
    
    text = unicode_normalize(name).lower().strip()
    
    # Remove URL prefixes
    text = re.sub(r'^https?://', '', text)
    text = re.sub(r'^www\.', '', text)
    
    # Remove URL domain extensions if it looks like a domain name
    text = DOMAIN_EXTENSIONS.sub('', text)
    
    # Replace common symbol replacements
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('+', ' plus ')
    
    # Replace non-alphanumeric with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    tokens = text.split()
    if not tokens:
        return ""
    
    # Remove trailing legal suffixes
    filtered_tokens = []
    for t in tokens:
        if t not in LEGAL_SUFFIXES:
            filtered_tokens.append(t)
            
    # If all tokens were stripped, keep original cleaned tokens
    result_tokens = filtered_tokens if filtered_tokens else tokens
    return " ".join(result_tokens)


def clean_address(address: Optional[str]) -> str:
    """
    Cleans and canonicalizes an address:
    - Normalizes abbreviations (rd -> road, st -> street, etc.)
    - Removes punctuation and excessive whitespace
    - Preserves digits (PIN/Zip codes, building numbers)
    """
    if not address or str(address).lower() == 'nan':
        return ""
    
    text = unicode_normalize(address).lower().strip()
    
    # Replace symbols
    text = text.replace('#', ' number ')
    text = text.replace('/', ' ')
    text = text.replace('-', ' ')
    text = text.replace(',', ' ')
    
    # Replace non-alphanumeric with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    tokens = text.split()
    normalized_tokens = [ADDRESS_ABBR.get(t, t) for t in tokens]
    return " ".join(normalized_tokens)


def extract_numbers(text: Optional[str]) -> Set[str]:
    """Extracts all digit sequences (PIN codes, house/street numbers) from text."""
    if not text or str(text).lower() == 'nan':
        return set()
    return set(re.findall(r'\b\d+\b', str(text)))


if __name__ == "__main__":
    # Test cases
    test_names = [
        "Maure Williams Colombier Inc",
        "Maure Wilblims Colombier Inc",
        "maurewilliamscolombier.com",
        "AT&T Services, LLC",
        "Société Générale S.A.R.L.",
        "Tata Consultancy Services Pvt. Ltd."
    ]
    print("--- Test Name Cleaning ---")
    for n in test_names:
        print(f"'{n}' -> '{clean_name(n)}'")
        
    test_addrs = [
        "85 Wayne Avenue, Ticonderoga, NY",
        "85 Wanye Avenue, Ticonderoga Townshiip, New York",
        "Wayne Ave, Ticonderoga Townshiip, New York",
        "Near SBI ATM, Sector 14, Gurgaon, 122001",
        "12 Rue de la Paix, 75002 Paris"
    ]
    print("\n--- Test Address Cleaning ---")
    for a in test_addrs:
        print(f"'{a}' -> '{clean_address(a)}' | Numbers: {extract_numbers(a)}")
