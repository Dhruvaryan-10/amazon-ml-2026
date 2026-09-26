"""Text normalisation for names and addresses (country-agnostic, handles Indic scripts + French)."""
import re
import unicodedata
from unidecode import unidecode

# --- native-script state names -> latin (hand-written, applied BEFORE transliteration) ---
NATIVE = {
    "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu", "महाराष्ट्र": "maharashtra",
    "दिल्ली": "delhi", "उत्तर प्रदेश": "uttar pradesh", "मध्य प्रदेश": "madhya pradesh",
    "राजस्थान": "rajasthan", "हरियाणा": "haryana", "बिहार": "bihar", "ગુજરાત": "gujarat",
    "পশ্চিমবঙ্গ": "west bengal", "తెలంగాణ": "telangana", "ఆంధ్ర ప్రదేశ్": "andhra pradesh",
    "കേരളം": "kerala", "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
}

# --- legal forms (US / India / France + transliterated Hindi) -> canonical ---
LEGAL = {
    "pvt": "private", "pvte": "private", "private": "private", "praivet": "private", "praivt": "private",
    "pra": "private", "li": "limited", "ltd": "limited", "limited": "limited", "limitted": "limited", "limited.": "limited",
    "llc": "llc", "llp": "llp", "elelpi": "llp", "inc": "inc", "incorporated": "inc",
    "corp": "corporation", "corporation": "corporation", "co": "company", "company": "company",
    "plc": "plc", "opc": "opc", "lp": "lp", "pllc": "pllc", "pc": "pc",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "snc": "snc",
    "sci": "sci", "scop": "scop", "societe": "societe", "ste": "societe", "gmbh": "gmbh",
}
SKEL_LEGAL = {"prwt": "private", "prwte": "private", "lmtd": "limited", "lmtt": "limited",
              "lmtdd": "limited", "lmttd": "limited", "lmt": "limited", "krprtn": "corporation", "kmpn": "company"}
HONORIFIC = {"m", "s", "ms", "mr", "mrs", "dr", "shri", "sri", "smt", "messrs", "the", "and", "et", "le", "la", "les"}

# --- address abbreviations (applied on tokens) ---
ADDR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "hwy": "highway", "pkwy": "parkway", "pky": "parkway",
    "pl": "place", "sq": "square", "ter": "terrace", "trl": "trail", "tr": "trail", "wy": "way",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "mt": "mount", "ft": "fort",
    "chem": "chemin", "rte": "route", "imp": "impasse", "fbg": "faubourg", "all": "allee",
    "bengaluru": "bangalore", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurugram": "gurgaon", "nagr": "nagar", "opp": "opposite", "nr": "near",
}
ADDR_DROP = {"door", "no", "number", "null", "none", "na", "nan", "cedex", "bp"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
# state names are replaced by a marker token so they can't collide with words like "in", "or", "me"
_STATE_RE = {
    c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b")
    for c, d in [("US", US_STATES), ("India", IN_STATES)]
}
_STATE_MAP = {"US": US_STATES, "India": IN_STATES}

_DOMAIN = re.compile(r"^(https?://)?(www\.)?([a-z0-9\-]+)\.(com|net|org|in|co\.in|co|biz|info|fr|us|io)(/\S*)?$")
_REPEAT = re.compile(r"([a-z])\1+")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_POBOX = re.compile(r"\bp\.?\s*o\.?\s*box\s*\d+")


def to_ascii(s: str) -> str:
    """Latin text is only accent-stripped; Indic scripts are transliterated and
    their doubled letters collapsed ("maarkettiNg" -> "marketing")."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for k, v in NATIVE.items():
        if k in s:
            s = s.replace(k, " " + v + " ")
    if s.isascii():
        return s.lower()
    out = []
    for w in s.split():
        latin = all(ord(ch) < 0x0370 for ch in w)          # Latin/accented Latin word
        a = unidecode(w).replace("oN", "o").replace("N", "n").lower()
        out.append(a if latin else _REPEAT.sub(r"\1", a))
    return " ".join(out)


def _clean_tokens(s: str) -> list:
    s = s.replace("&", " and ").replace("@", " ")
    s = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group().replace(".", ""), s)   # l.l.c. -> llc
    return [t for t in _NONALNUM.sub(" ", s).split() if t]


def skel(t: str) -> str:
    """Phonetic consonant skeleton, robust to transliteration:
    'sonftveyr'/'software' -> 'sftwr', 'prphekt'/'perfect' -> 'prfkt', 'piraivet'/'private' -> 'prwt'."""
    if t.isdigit():
        return t
    t = t.replace("ph", "f").replace("ck", "k").replace("c", "k").replace("q", "k")
    t = t.replace("x", "ks").replace("v", "w").replace("z", "s").replace("h", "")
    t = re.sub(r"[aeiouy]", "", t)
    return _REPEAT.sub(r"\1", t)


def consonants(tokens) -> str:
    return " ".join(x for x in (skel(t) for t in tokens) if x)


FR_LEGAL = {"cie": "company", "selarl": "selarl", "earl": "earl", "gaec": "gaec", "scea": "scea", "sca": "sca",
            "sel": "sel", "scp": "scp", "sem": "sem", "gie": "gie"}
# French street abbreviations (FR only: "st" is Saint in France but Street in the US)
FR_ADDR = {"st": "saint", "ste": "sainte", "r": "rue", "ch": "chemin", "chem": "chemin", "av": "avenue",
           "ave": "avenue", "bd": "boulevard", "blvd": "boulevard", "bld": "boulevard", "boul": "boulevard",
           "imp": "impasse", "pl": "place", "all": "allee", "rte": "route", "fg": "faubourg", "fbg": "faubourg",
           "sq": "square", "crs": "cours", "qu": "quai", "qua": "quai", "res": "residence", "lot": "lotissement",
           "zi": "zone", "za": "zone", "pte": "porte", "prom": "promenade", "sent": "sentier", "n": "", "no": "", "ndeg": "", "deg": ""}


def norm_name(raw: str, country: str = None) -> dict:
    s = to_ascii(raw).strip()
    s = re.sub(r"\bm\s*/\s*s\b", " ", s)          # "M/s"
    m = _DOMAIN.match(s.replace(" ", ""))          # "wilfordhancock.com" -> "wilfordhancock"
    is_domain = bool(m) and " " not in s.strip()
    if is_domain:
        s = m.group(3)
    toks = _clean_tokens(s)
    toks = [t for t in toks if t not in {"null", "none", "nan", "na"}]
    fr = country == "France"
    def legal_of(t):
        return LEGAL.get(t) or (FR_LEGAL.get(t) if fr else None) or (SKEL_LEGAL.get(skel(t)) if len(t) >= 5 else None)
    legal = sorted({legal_of(t) for t in toks if legal_of(t)})
    core = [t for t in toks if not legal_of(t) and t not in HONORIFIC]
    if not core:                                  # name was only legal words -> keep them
        core = [t for t in toks if t not in HONORIFIC] or toks
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_sorted": " ".join(sorted(core)),
        "name_nospace": "".join(core),            # "2827art" vs "27 28 art"
        "name_cons": consonants(core),
        "legal": " ".join(legal),
        "name_is_domain": is_domain,
    }


def norm_addr(raw: str, country: str) -> dict:
    s = to_ascii(raw)
    s = _POBOX.sub(" ", s)
    s = re.sub(r"\bc\s*/\s*o\b[^,]*,?", " ", s)          # "C/O Sunita Singh," -> removed
    # US zip / FR code postal (5 digits), IN PIN (6). Ignore a number that STARTS the
    # address: that is a house number ("01130 Regency Road"), not a postcode.
    postcodes = [m.group() for m in re.finditer(r"\b\d{5,6}\b", s)
                 if s[:m.start()].strip(" ,#-")]
    state = ""
    if country in _STATE_RE:
        s2 = s.replace(".", " ")
        found = _STATE_RE[country].findall(s2)
        if found:
            state = _STATE_MAP[country][found[-1]]
            s = _STATE_RE[country].sub(" ", s2)
    toks = _clean_tokens(s)
    toks = [t for t in toks if t not in ADDR_DROP]
    if country == "France":
        toks = [FR_ADDR.get(t, ADDR.get(t, t) if len(t) > 1 else t) for t in toks]
        toks = [t for t in toks if t]
    else:
        toks = [t if (len(t) == 1 and country != "US") else ADDR.get(t, t) for t in toks]
    toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in toks]   # "01130" -> "1130"
    # a lone 2-letter token that is a state code (e.g. "nc", "ka") -> state field
    codes = set(_STATE_MAP.get(country, {}).values())
    if not state:
        for t in reversed(toks):
            if len(t) == 2 and t in codes:
                state = t
                break
    toks = [t for t in toks if not (len(t) == 2 and t in codes)]
    pc = postcodes[-1] if postcodes else ""
    nums = sorted({n.lstrip("0") or "0" for n in re.findall(r"\d+", " ".join(toks)) if n != pc})
    words = [t for t in toks if not t.isdigit()]
    return {
        "addr_norm": " ".join(toks),
        "addr_sorted": " ".join(sorted(set(toks))),
        "addr_words": " ".join(sorted(set(words))),
        "addr_nums": " ".join(nums),
        "postcode": pc,
        "state": state,
    }


def normalize_df(df):
    """Adds normalised columns to a source dataframe (entity_id, business_name, business_address, country)."""
    import pandas as pd
    names = pd.DataFrame([norm_name(x, c) for x, c in zip(df["business_name"].values, df["country"].values)],
                         index=df.index)
    addrs = pd.DataFrame([norm_addr(a, c) for a, c in zip(df["business_address"].values,
                                                          df["country"].values)], index=df.index)
    return pd.concat([df, names, addrs], axis=1)


if __name__ == "__main__":
    tests = [
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
        ("Pvt. EFS Print Ventures Ltd.", "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ", "India"),
        ("M/s #southerneducational", "Door No 236 Floor Salarapuria Windsor No 3 Ulsoor Road, Bengaluru, ಕರ್ನಾಟಕ, Bangalore", "India"),
        ("Galaxy Solutions Pvt Ltd.", "NULL, MH, 4-7/1 To 14 Plot No. 15 Airport Road, Kolhapur", "India"),
        ("TRI-STATE CORNERSTONE BERTO  [LLC]", "1500 JUPITER RD, PO BOX 8832, ALLEN, TX", "US"),
        ("Tri-State Córnerstone Berto LLC", "Texas, # 609, Allen, 1500 Jupiter Road", "US"),
        ("2827art.com", "120 1/2 West Street, Worthington, Indiana", "US"),
        ("Fafloon Care Inc Center", "01130 REGENCY ROAD, ATL, GA", "US"),
        ("Boulangerie Dupont SARL", "12 bd de la République, 75011 Paris", "France"),
    ]
    for n, a, c in tests:
        print(f"\n{n} | {a}")
        print("  ", norm_name(n))
        print("  ", norm_addr(a, c))
