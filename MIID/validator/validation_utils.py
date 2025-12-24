# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

"""
Shared validation utilities for both miner and validator.
This module contains validation logic that should be identical in both components.
"""

import re
from typing import Dict, List, Tuple, Set
from datetime import datetime, timedelta
import bittensor as bt


def looks_like_address(address: str) -> bool:
    """
    Heuristic check if a string looks like a valid address.
    
    Requirements:
    - 30-300 characters (excluding whitespace/punctuation)
    - At least 20 letters
    - At least 2 commas
    - Numbers in at least one comma-separated section
    - No special characters: ` : % $ @ * ^ [ ] { } _ « »
    
    Args:
        address: The address string to validate
        
    Returns:
        True if address passes all heuristic checks, False otherwise
    """
    address = address.strip().lower()

    # Keep all letters (Latin and non-Latin) and numbers
    address_len = re.sub(r'[^\w]', '', address.strip(), flags=re.UNICODE)
    if len(address_len) < 30:
        return False
    if len(address_len) > 300:  # maximum length check
        return False

    # Count letters (both Latin and non-Latin)
    letter_count = len(re.findall(r'[^\W\d]', address, flags=re.UNICODE))
    if letter_count < 20:
        return False

    if re.match(r"^[^a-zA-Z]*$", address):  # no letters at all
        return False
    if len(set(address)) < 5:  # all chars basically the same
        return False
        
    # Has at least one digit in a comma-separated section
    address_for_number_count = address.replace('-', '').replace(';', '')
    sections = [s.strip() for s in address_for_number_count.split(',')]
    sections_with_numbers = []
    for section in sections:
        number_groups = re.findall(r"[0-9]+", section)
        if len(number_groups) > 0:
            sections_with_numbers.append(section)
    if len(sections_with_numbers) < 1:
        return False

    if address.count(",") < 2:
        return False
    
    # Check for special characters that should not be in addresses
    special_chars = ['`', ':', '%', '$', '@', '*', '^', '[', ']', '{', '}', '_', '«', '»']
    if any(char in address for char in special_chars):
        return False
    
    return True


# Global country name mapping
COUNTRY_MAPPING = {
    "korea, south": "south korea",
    "korea, north": "north korea",
    "cote d ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "the gambia": "gambia",
    "netherlands": "the netherlands",
    "holland": "the netherlands",
    "congo, democratic republic of the": "democratic republic of the congo",
    "drc": "democratic republic of the congo",
    "congo, republic of the": "republic of the congo",
    "burma": "myanmar",
    "bonaire": "bonaire, saint eustatius and saba",
    "usa": "united states",
    "us": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "britain": "united kingdom",
    "uae": "united arab emirates",
    "u.s.a.": "united states",
    "u.s.": "united states",
    "u.k.": "united kingdom",
    "north macedonia, the republic of": "north macedonia",
    "palestine, state of": "palestinian territory",
    "palestinian": "palestinian territory",
}


def extract_city_country(address: str, two_parts: bool = False) -> Tuple[str, str]:
    """
    Extract city and country from an address.
    
    Args:
        address: The address to extract from
        two_parts: If True, treat last two comma-separated segments as country
        
    Returns:
        Tuple of (city, country) - both strings, empty if not found
    """
    if not address:
        return "", ""

    address = address.lower()
    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 2:
        return "", ""

    # Determine country
    last_part = parts[-1]
    single_part_normalized = COUNTRY_MAPPING.get(last_part, last_part)
    
    country_checking_name = ''
    if two_parts and len(parts) >= 2:
        two_part_raw = f"{parts[-2]}, {parts[-1]}"
        two_part_normalized = COUNTRY_MAPPING.get(two_part_raw, two_part_raw)
        if two_part_raw != two_part_normalized:
            country_checking_name = two_part_normalized
            normalized_country = two_part_normalized
            used_two_parts_for_country = True

    if country_checking_name == '':
        country_checking_name = single_part_normalized
        normalized_country = single_part_normalized
        used_two_parts_for_country = False

    if not normalized_country:
        return "", ""

    # Try to find city using geonames (simplified version without geonames dependency)
    # For miner pre-validation, we just extract the city part
    exclude_count = 2 if used_two_parts_for_country else 1
    
    # Simple extraction: take the part before the country
    if len(parts) > exclude_count:
        # Get the part just before country
        city_candidate = parts[-(exclude_count + 1)]
        # Remove numbers and get first word
        words = city_candidate.split()
        for word in words:
            if not any(char.isdigit() for char in word) and len(word) > 2:
                return word, normalized_country
    
    return "", normalized_country


def validate_address_region(generated_address: str, seed_address: str) -> bool:
    """
    Validate that generated address has correct region from seed address.
    
    Special handling for disputed regions: Luhansk, Crimea, Donetsk, West Sahara
    
    Args:
        generated_address: The generated address to validate
        seed_address: The seed address to match against
        
    Returns:
        True if region is valid, False otherwise
    """
    if not generated_address or not seed_address:
        return False
    
    seed_lower = seed_address.lower()
    
    # Special handling for Western Sahara
    WESTERN_SAHARA_CITIES = [
        "laayoune", "dakhla", "boujdour", "es semara", "sahrawi", "tifariti", "aousserd"
    ]
    if seed_lower in ["west sahara", "western sahara"]:
        gen_lower = generated_address.lower()
        return any(city in gen_lower for city in WESTERN_SAHARA_CITIES)
    
    # Other special regions
    OTHER_SPECIAL_REGIONS = ["luhansk", "crimea", "donetsk", "macau"]
    if seed_lower in OTHER_SPECIAL_REGIONS:
        return seed_lower in generated_address.lower()
    
    # Extract city and country
    gen_city, gen_country = extract_city_country(generated_address, two_parts=(',' in seed_address))
    seed_address_lower = seed_address.lower()
    seed_address_mapped = COUNTRY_MAPPING.get(seed_address.lower(), seed_address.lower())
    
    if not gen_city or not gen_country:
        return False
    
    # Check if either city or country matches
    city_match = gen_city and seed_address_lower and gen_city == seed_address_lower
    country_match = gen_country and seed_address_lower and gen_country == seed_address_lower
    mapped_match = gen_country and seed_address_mapped and gen_country == seed_address_mapped
    
    return city_match or country_match or mapped_match


def check_dob_categories(variations: Dict[str, List[List[str]]], identity_list: List[List[str]]) -> float:
    """
    Check if DOB variations meet required categories.
    
    Required categories (must have at least one variation in EACH):
    - ±1 day
    - ±3 days
    - ±30 days
    - ±90 days
    - ±365 days
    - Year+Month only (YYYY-MM format)
    
    Args:
        variations: Dictionary mapping names to [name, dob, address] variation lists
        identity_list: List of [name, dob, address] seed identities
        
    Returns:
        Score from 0.0 to 1.0 (1.0 = all categories present for all names)
    """
    ranges = [1, 3, 30, 90, 365]
    total_ranges = len(ranges) + 1  # +1 for year_month
    
    name_scores = []
    
    for name_idx, name in enumerate(variations.keys()):
        if name not in variations or len(variations[name]) < 1 or name_idx >= len(identity_list):
            continue
            
        identity = identity_list[name_idx]
        if len(identity) < 2 or not identity[1]:
            continue
            
        seed_dob = identity[1]
        all_variations = variations[name]
        dob_variations = [var[1] for var in all_variations if len(var) > 1 and var[1]]
        
        if not dob_variations:
            continue
        
        name_found_ranges = set()
        try:
            seed_date = datetime.strptime(seed_dob, "%Y-%m-%d")
            
            for dob_var in dob_variations:
                if not dob_var:
                    continue
                    
                try:
                    var_date = datetime.strptime(dob_var, "%Y-%m-%d")
                    day_diff = abs((var_date - seed_date).days)
                    
                    if day_diff <= 1:
                        name_found_ranges.add(1)
                    elif day_diff <= 3:
                        name_found_ranges.add(3)
                    elif day_diff <= 30:
                        name_found_ranges.add(30)
                    elif day_diff <= 90:
                        name_found_ranges.add(90)
                    elif day_diff <= 365:
                        name_found_ranges.add(365)
                        
                except ValueError:
                    try:
                        year_month = datetime.strptime(dob_var, "%Y-%m")
                        if (seed_date.year == year_month.year and 
                            seed_date.month == year_month.month):
                            name_found_ranges.add("year_month")
                    except ValueError:
                        continue
            
            name_score = len(name_found_ranges) / total_ranges if total_ranges > 0 else 0.0
            name_scores.append(name_score)
                
        except ValueError:
            name_scores.append(0.0)
    
    if name_scores:
        return sum(name_scores) / len(name_scores)
    else:
        return 0.0


def check_address_quality(variations: Dict[str, List[List[str]]], identity_list: List[List[str]]) -> float:
    """
    Check address quality using validation functions.
    
    Args:
        variations: Dictionary mapping names to [name, dob, address] variation lists
        identity_list: List of [name, dob, address] seed identities
        
    Returns:
        Score from 0.0 to 1.0 (1.0 = all addresses pass validation)
    """
    total_addresses = 0
    passed_addresses = 0
    
    for name_idx, name in enumerate(variations.keys()):
        if name not in variations or len(variations[name]) < 1 or name_idx >= len(identity_list):
            continue
            
        identity = identity_list[name_idx]
        if len(identity) < 3 or not identity[2]:
            continue
            
        seed_address = identity[2]
        all_variations = variations[name]
        address_variations = [var[2] for var in all_variations if len(var) > 2 and var[2] and var[2].strip()]
        
        for addr in address_variations:
            total_addresses += 1
            
            if looks_like_address(addr):
                if validate_address_region(addr, seed_address):
                    passed_addresses += 1
    
    if total_addresses > 0:
        return passed_addresses / total_addresses
    else:
        return 1.0  # No addresses to check

