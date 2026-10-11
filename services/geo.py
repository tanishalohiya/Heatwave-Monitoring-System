import json
import difflib


def load_district_geojson(path):
    """
    Load a district boundary GeoJSON file from disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_district_name(name):
    """
    Lowercase, strip whitespace, and drop common prefixes/suffixes so
    minor formatting differences ("Sri Ganganagar" vs "Ganganagar",
    "Dist. Jaipur" vs "Jaipur") don't block a match.
    """
    if name is None:
        return ""
    name = str(name).strip().lower()
    for prefix in ("sri ", "shri ", "dist. ", "district "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("district", "").strip()
    return " ".join(name.split())


def build_name_lookup(geojson, name_property="Dist_Name"):
    """
    Returns dict: normalized_geojson_name -> original geojson Dist_Name value
    """
    lookup = {}
    for feature in geojson["features"]:
        raw_name = feature["properties"].get(name_property)
        lookup[normalize_district_name(raw_name)] = raw_name
    return lookup


def match_district_names(dataset_district_names, geojson, name_property="Dist_Name",
                          fuzzy_cutoff=0.75):
    """
    Match each district name from your dataset to a name in the geojson.

    Tries, in order:
      1. Exact match after normalization
      2. Fuzzy match (handles spelling variants like Jalor/Jalore,
         Dhaulpur/Dholpur, Chittaurgarh/Chittorgarh)

    Returns:
        matched: dict {dataset_name: geojson_Dist_Name}
        unmatched: list of dataset names with no confident match
    """
    lookup = build_name_lookup(geojson, name_property)
    normalized_keys = list(lookup.keys())

    matched = {}
    unmatched = []

    for name in dataset_district_names:
        key = normalize_district_name(name)

        if key in lookup:
            matched[name] = lookup[key]
            continue

        close = difflib.get_close_matches(key, normalized_keys, n=1, cutoff=fuzzy_cutoff)
        if close:
            matched[name] = lookup[close[0]]
        else:
            unmatched.append(name)

    return matched, unmatched