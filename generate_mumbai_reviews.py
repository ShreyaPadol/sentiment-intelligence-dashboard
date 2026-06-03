"""
generate_mumbai_reviews.py
Generates statistically realistic synthetic reviews for 163 Mumbai petrol pumps,
seeded from their actual aggregate ratings and review counts.
Produces: mumbai_petrol_reviews.csv
"""

import random
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Review text templates by sentiment tier ────────────────────────────────────

POSITIVE_TEMPLATES = [
    "Very clean and well-maintained station. Staff is polite and helpful. No waiting time.",
    "Quick service, honest billing. The attendant gave exact change without any issues.",
    "One of the best petrol pumps in {zone}. Always get full quantity fuel here.",
    "Staff is well-behaved and the meter is always reset to zero before filling. Trust them completely.",
    "Clean washroom facilities available. Air pump also works well. Good overall experience.",
    "Never had any issue with billing. UPI and card payments both accepted without fuss.",
    "Great location, easy entry and exit. No long queues even in peak hours.",
    "Genuinely good service by {brand} staff. They fill fuel properly and don't try to cheat.",
    "The fuel quality is good — my mileage has been consistent since I started using this pump.",
    "Fast service. Came during morning rush, waited only 2 minutes. Well organized queue.",
    "Very trustworthy pump. The display shows the correct reading and staff is courteous.",
    "Superb cleanliness. Washrooms are clean, premises well swept. Fuel is of good quality.",
    "Efficient staff and no overcharging. Even gave receipt without asking. Highly recommended.",
    "This {brand} pump is my regular stop. Never had any complaint in years of use.",
    "Smooth experience. PhonePe and GPay accepted. Staff reset the meter properly.",
    "Excellent service. Even air inflation is available for free here. Staff very helpful.",
    "Well-lit, secure and clean. Staff follow proper protocols. Trustworthy {brand} outlet.",
    "Good pump with dedicated lanes for two-wheelers and four-wheelers. Efficient queue management.",
    "Regularly visit this pump. They always give correct quantity. No manipulation ever noticed.",
    "Staff is professional and billing is transparent. Digital receipts available on request.",
]

NEUTRAL_TEMPLATES = [
    "Average experience. Service is okay but the queue can get long during peak hours.",
    "Decent pump. Staff attitude is alright. Nothing exceptional but no major complaints either.",
    "Okay service. Waiting time is a bit high on weekends. Billing seems correct.",
    "The pump is fine. Sometimes the UPI terminal doesn't work — have to pay cash.",
    "Mixed feelings. Staff is helpful on some visits, indifferent on others. Quality of fuel seems okay.",
    "Not the cleanest pump but not dirty either. Service is average. Gets job done.",
    "Typical {brand} outlet. Nothing special to note. Standard service, standard wait time.",
    "Location is convenient but parking for 4-wheelers is tricky. Otherwise okay.",
    "Some days quick, some days slow depending on the shift. Fuel quality seems fine.",
    "Moderate experience. Had to wait 10 minutes on a busy day. Staff was polite though.",
    "Regular pump. Nothing wrong, nothing outstanding. Billing correct. Accepts digital payments.",
    "The pump is okay for routine filling. Wouldn't go out of my way for it though.",
    "Service is average. Sometimes they are short-staffed which causes delays. Fuel quality fine.",
    "Had to ask twice to get the meter reset. Otherwise billing was correct. Neutral experience.",
    "Decent {brand} station. Air pump not always functional. Rest of the service is fine.",
]

NEGATIVE_TEMPLATES = [
    "Staff is very rude. Asked for receipt and was ignored. Will not return to this pump.",
    "Meter not reset to zero before filling. When confronted, staff was dismissive.",
    "Overcharged by Rs 50. The display was manipulated. Complete fraud happening here.",
    "Long wait time even when there are only 2-3 vehicles ahead. Poor management.",
    "Dirty washrooms. The premises smell bad. Staff doesn't bother to clean.",
    "UPI terminal always down. Have to carry exact cash. Very inconvenient.",
    "Suspicious fuel quality. My mileage dropped significantly after filling from this pump.",
    "No CNG available even though it's listed as a CNG station. Wasted my trip.",
    "Staff argues with customers when pointed out discrepancy in meter reading. Avoid this pump.",
    "Very bad experience. They didn't give receipt. The meter reading was suspicious.",
    "Terrible service. Waited 20 minutes for fuel. Only one nozzle working. Pathetic management.",
    "Staff attitude is horrible. Rude, arrogant, not helpful at all. Escalated complaint to company.",
    "Billing fraud detected — was charged for 5 litres more than actual fill. Be careful.",
    "Scam alert! The meter starts from non-zero position. Don't visit this {brand} pump.",
    "No CNG available since last two weeks. No information on when it will resume. Terrible.",
    "Fuel spilled on my vehicle. Staff showed no concern or apology. Very irresponsible.",
    "The air machine is broken and has been for months. No one bothers to fix anything here.",
    "Rude attendant refused to give bill. When I insisted, he became aggressive. Very unsafe.",
    "Cheating happening openly. Please avoid this pump. They will manipulate the meter.",
    "My vehicle engine gave trouble after filling petrol here. Fuel quality is suspect.",
    "Queue management is non-existent. Took 30 minutes to fill fuel. Completely disorganized.",
    "Card machine not working, UPI not working, cash only. What year is it? Pathetic setup.",
    "Staff was smoking near the fuel area. Major safety hazard. Complained but no action.",
    "Dishonest staff. Gave less fuel than charged for. Totally untrustworthy pump.",
    "Very dirty. Garbage all around. Washroom is disgusting. Avoid if possible.",
]

CNG_EXTRA_NEGATIVE = [
    "No CNG available again today. This is the third time this week. Completely unreliable.",
    "CNG station shows open but compressor is down. Drove 5 km for nothing. Very frustrating.",
    "CNG pressure is always low here. Takes 20 minutes to fill what should take 5. Pathetic.",
    "No CNG since morning. Staff says will resume by evening. No accountability at all.",
    "CNG queue is always 30-40 vehicles long. Need better management of CNG filling.",
]

CNG_POSITIVE = [
    "CNG always available. Fast filling and honest billing. Best CNG pump in the area.",
    "Good CNG pressure. Fills quickly. Staff is helpful. No complaints.",
    "Reliable CNG supply. Never found it closed or out of stock. Trustworthy station.",
    "CNG availability is consistent here. Queue is managed well. Good experience.",
]

BRAND_MAP = {
    "hindustan petroleum": "HP", "hp ": "HP", "hpcl": "HP",
    "bharat petroleum": "BPCL", "bpcl": "BPCL",
    "indianoil": "IndianOil", "indian oil": "IndianOil", "iocl": "IndianOil",
    "mahanagar gas": "Mahanagar Gas", "mgl": "Mahanagar Gas",
    "essar": "Essar", "shell": "Shell", "nayara": "Nayara",
    "reliance": "Reliance",
}


def extract_brand(title):
    t = str(title).lower()
    for key, val in BRAND_MAP.items():
        if key in t:
            return val
    return "Other"


def assign_zone(lat, lng):
    try:
        lat, lng = float(lat), float(lng)
    except Exception:
        return "Mumbai"
    if lat < 18.970:
        return "South Mumbai"
    elif lat < 19.050:
        return "Central Mumbai"
    elif lat < 19.150:
        return "Western Suburbs" if lng < 72.880 else "Eastern Suburbs"
    else:
        return "North Suburbs"


def generate_reviews_for_station(row, n_reviews):
    """Generate n_reviews for a single station based on its aggregate rating."""
    rating = row.get("totalScore", 3.5)
    if pd.isna(rating):
        rating = 3.5
    rating = float(rating)

    is_cng = "compressed" in str(row.get("categoryName", "")).lower()
    brand = extract_brand(row.get("title", ""))
    zone = assign_zone(row.get("lat", row.get("latitude_clean")),
                       row.get("lng", row.get("longitude_clean")))
    station_name = str(row.get("title", "Unknown Station"))

    # Derive sentiment mix from real rating
    # Rating ≥ 4.0: ~70% pos, 20% neu, 10% neg
    # Rating 3.0–3.9: ~35% pos, 40% neu, 25% neg
    # Rating < 3.0: ~10% pos, 20% neu, 70% neg
    if rating >= 4.2:
        pos_p, neu_p, neg_p = 0.75, 0.18, 0.07
    elif rating >= 3.7:
        pos_p, neu_p, neg_p = 0.60, 0.28, 0.12
    elif rating >= 3.3:
        pos_p, neu_p, neg_p = 0.40, 0.38, 0.22
    elif rating >= 2.8:
        pos_p, neu_p, neg_p = 0.20, 0.30, 0.50
    else:
        pos_p, neu_p, neg_p = 0.08, 0.12, 0.80

    # Star distribution mirroring the rating
    def random_star(sentiment):
        if sentiment == "Positive":
            return random.choices([4, 5], weights=[0.35, 0.65])[0]
        elif sentiment == "Neutral":
            return random.choices([3, 4], weights=[0.7, 0.3])[0]
        else:
            return random.choices([1, 2], weights=[0.55, 0.45])[0]

    def pick_template(sentiment):
        if sentiment == "Positive":
            if is_cng and random.random() < 0.3:
                pool = CNG_POSITIVE
            else:
                pool = POSITIVE_TEMPLATES
        elif sentiment == "Neutral":
            pool = NEUTRAL_TEMPLATES
        else:
            if is_cng and random.random() < 0.4:
                pool = CNG_EXTRA_NEGATIVE
            else:
                pool = NEGATIVE_TEMPLATES
        t = random.choice(pool)
        return t.replace("{brand}", brand).replace("{zone}", zone)

    # Random date in past 18 months
    def random_date():
        days_back = random.randint(1, 548)
        return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Author names pool
    author_names = [
        "Rahul Sharma", "Priya Mehta", "Amit Patil", "Suresh Nair", "Kavita Joshi",
        "Mohammed Ansari", "Deepak Gupta", "Anjali Desai", "Vijay Kumar", "Sneha Rao",
        "Rajesh Iyer", "Neha Pandey", "Sanjay Verma", "Pooja Shah", "Arun Tiwari",
        "Meera Pillai", "Rohit Malhotra", "Divya Chavan", "Kiran Sawant", "Ashok More",
        "Sunita Bhosle", "Nikhil Thakur", "Leela Nair", "Ganesh Parab", "Usha Kadam",
        "Pankaj Dubey", "Rekha Jain", "Sunil Kamble", "Anita Kulkarni", "Vikram Shinde",
    ]

    reviews = []
    sentiments = random.choices(
        ["Positive", "Neutral", "Negative"],
        weights=[pos_p, neu_p, neg_p],
        k=n_reviews
    )

    for i, sentiment in enumerate(sentiments):
        text = pick_template(sentiment)
        star = random_star(sentiment)
        review_id = hashlib.md5(f"{station_name}{i}{text}".encode()).hexdigest()

        reviews.append({
            "review_id":       review_id,
            "station_name":    station_name,
            "station_address": row.get("address", ""),
            "station_rating":  rating,
            "station_reviews": row.get("reviewsCount", 0),
            "brand":           brand,
            "category":        "CNG Station" if is_cng else "Petrol Pump",
            "zone":            zone,
            "lat":             row.get("lat", row.get("latitude_clean")),
            "lng":             row.get("lng", row.get("longitude_clean")),
            "author_name":     random.choice(author_names),
            "rating":          star,
            "text":            text,
            "date_iso":        random_date(),
            "fuel_status":     row.get("Fuel_Category_Status", ""),
        })

    return reviews


def generate_all(pumps_path="mumbai_petrol_pumps.xlsx",
                 out_path="mumbai_petrol_reviews.csv",
                 reviews_per_station_cap=30):
    df = pd.read_excel(pumps_path)
    print(f"Loaded {len(df)} stations from {pumps_path}")

    all_reviews = []
    for _, row in df.iterrows():
        rc = row.get("reviewsCount", 0)
        if pd.isna(rc):
            rc = 0
        # Scale real review count to a manageable synthetic number
        # Use log-scaled cap: a station with 10k reviews gets ~28, one with 100 gets ~10
        n = min(reviews_per_station_cap, max(5, int(np.log1p(float(rc)) * 3)))
        reviews = generate_reviews_for_station(row.to_dict(), n)
        all_reviews.extend(reviews)

    result = pd.DataFrame(all_reviews)
    result.to_csv(out_path, index=False)
    print(f"Generated {len(result):,} reviews for {df.shape[0]} stations → {out_path}")
    return result


if __name__ == "__main__":
    generate_all()
