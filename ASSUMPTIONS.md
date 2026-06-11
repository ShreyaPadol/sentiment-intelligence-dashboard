# Methodology, Assumptions & Brand Classification Rules

*Last updated: 2026-06-11. Update alongside any changes to `sentiment_engine.py`, `scrape_mumbai_reviews.py`, or `app.py`.*

---

## 1. Data Sources

| Dataset | File | Format | Rows | Notes |
|---|---|---|---|---|
| Mumbai Petrol Pump Reviews | `mumbai_petrol_reviews.csv` | Review-level (raw) | 1,595 | Individual Google Maps reviews; pre-NLP |
| Mumbai Petrol Pump NLP Output | `mumbai_petrol_reviews_nlp.csv` | Review-level (enriched) | 1,595 | Adds sentiment, star, issue, quality columns |
| Mumbai Sentiment Analysis Export | `mumbai_sentiment_analysis.csv` | Review-level (labelled) | 1,595 | Final export from Data Export tab |
| Mumbai Station-Level Aggregate | `mumbai_petrol_pumps.xlsx` | Station-level | 168 | Aggregate Google ratings; no review text |
| Pune Retail Outlets | `Pune_Retail_outlet (1).xlsx` | Review-level | 568 | Individual review text; sourced separately |
| Mumbai Boundary | `mumbai_boundary.geojson` | GeoJSON polygon | 1 | Rendered as outline on the map tab |

---

## 2. Brand Classification — Rules & Assumptions

### 2.1 Why Brand Is Absent from Google Maps Data

Google Maps does not display the oil marketing company (OMC) brand (e.g. HPCL, BPCL, IndianOil) as a structured field. The data collected from Google Maps contains:
- **Station name** — a free-text string entered by the station owner or auto-filled by Google (e.g. "Hindustan Petroleum Corporation Limited", "HP Petrol Pump", "BPCL Fuel Station")
- **Address** — full street address
- **Category** — Google Places category ("Petrol pump", "Compressed natural gas station")
- **Ratings and review counts** — aggregate statistics

Brand is therefore **inferred** from the station name using string matching rules.

---

### 2.2 Brand Extraction Algorithm

**Source:** `scrape_mumbai_reviews.py` → function `extract_brand(title)`

```
BRAND_MAP = {
    "hindustan petroleum": "HP",
    "hp ":                 "HP",
    "hpcl":                "HP",
    "bharat petroleum":    "BPCL",
    "bpcl":                "BPCL",
    "indianoil":           "IndianOil",
    "indian oil":          "IndianOil",
    "iocl":                "IndianOil",
    "mahanagar gas":       "Mahanagar Gas",
    "mgl":                 "Mahanagar Gas",
    "essar":               "Essar",
    "shell":               "Shell",
    "nayara":              "Nayara",
    "reliance":            "Reliance",
}

def extract_brand(title):
    t = title.lower()
    for key, val in BRAND_MAP.items():
        if key in t:
            return val
    return "Other"
```

**Logic:**
1. The station name (title) is lowercased.
2. Each key in `BRAND_MAP` is checked as a substring.
3. The first matching key's value is returned as the brand.
4. If no key matches, the station is labelled **"Other"** — typically independent or unbranded stations.

---

### 2.3 Keyword Matching Rules and Rationale

| Keyword(s) | Assigned Brand | Rationale |
|---|---|---|
| `hindustan petroleum`, `hpcl`, `hp ` | **HP** | All official naming variants of HPCL. Note: `"hp "` includes a trailing space to avoid false matches with abbreviated words ending in "hp" (e.g. horsepower). |
| `bharat petroleum`, `bpcl` | **BPCL** | Full name and acronym of Bharat Petroleum Corporation Ltd. |
| `indianoil`, `indian oil`, `iocl` | **IndianOil** | Covers one-word and two-word spellings; official acronym IOCL also matched. |
| `mahanagar gas`, `mgl` | **Mahanagar Gas** | Mumbai's primary CNG distributor; most CNG-only stations appear under this name. |
| `shell` | **Shell** | International brand; small number of outlets in South and Central Mumbai. |
| `essar` | **Essar** | Legacy brand; rebranded to Nayara Energy in 2017, but some station names retain "Essar". |
| `nayara` | **Nayara** | Post-rebranding name of Essar retail network. |
| `reliance` | **Reliance** | Reliance Industries petroleum retail outlets. |
| *(no match)* | **Other** | Stations whose names do not contain any of the above strings. This includes: independent pump operators, ambiguous names, or stations using local names only. |

---

### 2.4 Address Validation for Brand Mapping

To reduce misclassification risk, the following cross-validation steps were applied during data collection and review:

1. **Address consistency check:** Stations labelled "HP" were spot-checked to confirm the address contains references consistent with HPCL dealerships (e.g. "Hindustan Petroleum" visible in the full street-level name on Google Maps).

2. **Coordinate plausibility:** Each station's latitude and longitude were verified to fall within Mumbai city limits (roughly 18.89°N–19.28°N, 72.77°E–73.00°E). Stations outside this bounding box were excluded from analysis.

3. **Category cross-reference:** Stations classified as "Mahanagar Gas" were verified against the Google Places category `"Compressed natural gas station"` — CNG stations should be MGL or affiliated; mismatches were flagged.

4. **Review count baseline:** Stations with fewer than 10 reviews were flagged as low-evidence and do not contribute to brand-level aggregates in a statistically meaningful way.

5. **Manual spot audit:** A random sample of 30 stations across all brands was manually cross-checked against publicly available dealer locators (HPCL, BPCL, IndianOil dealer search portals) to validate brand assignment accuracy. No systemic errors were found in the sample.

---

### 2.5 Known Limitations of Brand Mapping

- **Rebranded stations:** Some Essar outlets have been rebranded to Nayara but may still appear on Google Maps under the old name. These would be classified as "Essar" rather than "Nayara" by the current rules.
- **Franchise name variations:** Individual franchise owners sometimes register stations under personal or local names (e.g. "Ramesh Fuel Centre"). These cannot be matched and fall into "Other".
- **No official dealer registry cross-join:** The brand labels are derived solely from station names. A more robust approach would join against the official OMC dealer locator databases (HPCL, BPCL, IndianOil portals expose dealer lists). This was not implemented to avoid dependency on external APIs.
- **Brand distribution in dataset:**
  - HP: 596 reviews across the largest station footprint
  - BPCL: 389 reviews
  - IndianOil: 301 reviews
  - Mahanagar Gas: 227 reviews
  - Other: 72 reviews
  - Shell: 10 reviews (limited footprint in Mumbai)

---

## 3. Zone Assignment Rules

**Source:** `scrape_mumbai_reviews.py` → function `assign_zone(lat, lng)`

Zones are assigned by WGS-84 coordinate thresholds approximating Mumbai's colloquial geographic divisions:

| Zone | Latitude Condition | Longitude Condition | Key Areas |
|---|---|---|---|
| South Mumbai | lat < 18.970 | — | Colaba, Fort, Churchgate, Worli, Byculla |
| Central Mumbai | 18.970 ≤ lat < 19.050 | — | Dadar, Sion, Kurla, Chembur |
| Western Suburbs | 19.050 ≤ lat < 19.150 | lng < 72.880 | Bandra, Juhu, Andheri, Versova |
| Eastern Suburbs | 19.050 ≤ lat < 19.150 | lng ≥ 72.880 | Ghatkopar, Vikhroli, Mulund, Bhandup |
| North Suburbs | lat ≥ 19.150 | — | Goregaon, Malad, Kandivali, Borivali |

**Caveat:** These boundaries are coordinate-based approximations and do not follow official BMC ward boundaries. For administrative-precision work, a spatial join against Mumbai ward GeoJSON would be required.

---

## 4. Sentiment Analysis

### 4.1 BERT Model

- **Model:** `nlptown/bert-base-multilingual-uncased-sentiment` (HuggingFace)
- Trained on Amazon / Yelp / TripAdvisor reviews in 6 languages; effective for short Indian-English and code-switched (Hinglish) reviews.
- Outputs a 1–5 star prediction mapped as:
  - Stars 1–2 → **Negative**
  - Star 3 → **Neutral**
  - Stars 4–5 → **Positive**

### 4.2 Hard Negative Override

Reviews predicted as Positive are overridden to Negative if they contain high-signal fraud or complaint phrases (`fraud`, `scam`, `cheating`, `no cng`, `overcharg`, `short fill`, etc.). This corrects the model's tendency to misclassify code-switched complaints.

### 4.3 Confidence Score

The model returns a probability distribution over 5 classes. Confidence is defined as the probability mass of the predicted class. Scores below 0.35 indicate ambiguity.

---

## 5. Issue Classification

**Source:** `sentiment_engine.py`

- Keyword-based scoring across 17 issue categories (Meter Tampering & Fraud, Staff Behaviour, Fuel Short Filling, Waiting Time & Queue, Payment Methods, CNG Availability, Fuel Quality, Cleanliness & Hygiene, Safety Concern, Facility Maintenance, Billing Issue, Staff Helpfulness, Air/Tyre/Nitrogen, Operating Hours, Traffic & Accessibility, Amenities & ATM, Pricing).
- Each review is assigned the category with the highest keyword match score.
- Multi-label tagging (`issue_tags`) retains the top-2 categories for richer downstream filtering.
- Reviews with zero keyword matches → **"Other"**.

---

## 6. Review Date Reconstruction

Google Maps returns relative dates (e.g. "4 months ago", "a year ago", "3 years ago"). These are reconstructed to calendar months relative to the data collection date (June 2026):

| Relative String | Parsed Offset |
|---|---|
| "a week ago", "X weeks ago" | 0 months (current month) |
| "a month ago" | 1 month back |
| "X months ago" | X months back |
| "a year ago" | 12 months back |
| "X years ago" | X × 12 months back |
| "Edited …" prefix | Stripped; underlying date parsed as above |

This reconstruction is approximate (±1 month). It enables time-series trend analysis and ARIMA/SARIMA forecasting but should not be treated as an exact timestamp.

---

## 7. Output Files

| File | Description |
|---|---|
| `mumbai_petrol_reviews.csv` | Raw scraped reviews (pre-NLP). Contains: review ID, station metadata, reviewer name, star rating, review text, relative and ISO date, collection timestamp. |
| `mumbai_petrol_reviews_nlp.csv` | NLP-enriched reviews. Adds: sentiment label, model confidence, predicted star rating, primary issue category, multi-label issue tags, review quality score. |
| `mumbai_sentiment_analysis.csv` | Final labelled export (renamed columns for readability). Equivalent to the NLP file with human-readable column headers. |
| `scraped_reviews/` | Per-station JSON files from the scraper. Each file = one station's raw reviews. Basis for building the merged CSV. |

All files can be downloaded directly from the **Data Export** tab in the dashboard.

---

## 8. Known Data Gaps

1. No ward-level administrative boundaries used for zone assignment — coordinate thresholds are approximations.
2. Brand mapping relies solely on station name strings; no cross-join with OMC dealer registries.
3. Review dates are relative, not absolute — time-series analysis carries ±1-month uncertainty.
4. CNG availability at multi-fuel petrol stations is unknown (the dataset only flags dedicated CNG stations).
5. Stations with fewer than 10 Google reviews are statistically unreliable for individual-station conclusions.
6. No demographic data on reviewers; reviewer bias (e.g. frequent vs. one-time customers) cannot be controlled for.

---

## 9. Station-Level (Aggregate) Analysis

### 9.1 Sentiment Thresholds (Rating-Derived, Mumbai `mumbai_petrol_pumps.xlsx`)

| Rating Band | Sentiment Label | Rationale |
|---|---|---|
| ≤ 2.5 ★ | Negative | Consistently poor customer experience |
| 2.6 – 3.5 ★ | Neutral | Mixed or average performance |
| > 3.5 ★ | Positive | Generally satisfactory |

### 9.2 Stations Excluded

- 5 rows with `NaN` totalScore → labelled "Unknown"; excluded from all sentiment charts.
- 2 rows with null lat/lng → excluded from the geographic map only.

---

## 10. Validation Against Pune Baseline

| Metric | Pune (review-level) | Mumbai (station-level) |
|---|---|---|
| Sentiment model | BERT on text | Rating threshold |
| Issue categories | Keyword-NLP | Not available |
| Time-series | Via date column | Not available |
| Geographic map | Available | Full map view |

**Key limitation:** Pune and Mumbai aggregate results are not directly comparable because different sentiment methods are used.
