"""
Generates all six source files for the health insurance lakehouse project.

One script, not six, because the files must be referentially consistent:
every policy points at a real member, every claim at a real policy, and no
claim is dated after its policy lapsed.

Run:  python generate_source_data.py
Out:  ./output/*.csv  (6 files)

--------------------------------------------------------------------------
RULES PLANTED ON PURPOSE  (go and find these later - do not peek at this
list while you are analysing)
--------------------------------------------------------------------------
R1  LHC loading: 2% per year of age over 30 at join, capped at 70%.
R2  Acquisition channel correlates with age.
R3  State follows Australian population weights.
R4  Premiums rise every 1 April. Lapse rate spikes in April and May,
    hardest in Gold tier and in members with 4+ years tenure.
R5  Where a retention offer is made, roughly a third of at-risk members
    downgrade instead of lapsing.
R6  Extras-only members with zero claims in 12 months lapse ~3x more.
R7  An UNRESOLVED complaint raises lapse risk sharply for 90 days.
    A resolved complaint barely moves it. Resolution is what matters.
R8  Comparison-site and broker members churn faster but cost less to
    acquire. The ranking flips once you look at 3-year value.
R9  18-29s churn more and claim less. Deliberately ambiguous whether
    they are worth acquiring.
R10 Gold Family claims ratio deteriorates in WA and SA.
R11 Waiting periods suppress first-year claims. Anyone computing a
    claims ratio without excluding first-year policies gets it wrong.
    This is a trap. It is meant to be.

--------------------------------------------------------------------------
FAULTS PLANTED ON PURPOSE  (you will clean these in the Silver layer)
--------------------------------------------------------------------------
F1  state written three ways: VIC / Vic / Victoria
F2  gender coded six ways: M / F / Male / Female / U / blank
F3  dates in three formats across files
F4  ~3% of join_date missing
F5  40 near-duplicate members: trailing spaces, odd casing
F6  member_id zero-padded in the claims system, not in policy admin.
    Naive joins silently drop rows. This is the classic one.
F7  annual_premium stored as text with $ and thousands separators
F8  negative benefit_paid values: legitimate reversals, do not delete
F9  ~30 orphan claims pointing at policies that do not exist
F10 a few service dates in the future
F11 a few extras claims where benefit_paid exceeds charged_amount
"""

import os
import numpy as np
import pandas as pd

# ================================================================ settings
SEED = 42                 # change this and the whole dataset changes
N_MEMBERS = 10_000
OUTDIR = "output"

HISTORY_START = pd.Timestamp("2023-07-01")
HISTORY_END = pd.Timestamp("2026-06-30")

rng = np.random.default_rng(SEED)
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================ 1. PRODUCTS
def build_products():
    """One row per product per pricing period. Prices rise every 1 April."""
    base = [
        # code, name, tier, cover_type, premium at 2023 prices (single, $0 excess)
        ("HOSP_BASIC",       "Basic Hospital",        "Basic",       "hospital", 1100),
        ("HOSP_BRONZE",      "Bronze Hospital",       "Bronze",      "hospital", 1400),
        ("HOSP_BRONZE_PLUS", "Bronze Plus Hospital",  "Bronze Plus", "hospital", 1600),
        ("HOSP_SILVER",      "Silver Hospital",       "Silver",      "hospital", 1900),
        ("HOSP_SILVER_PLUS", "Silver Plus Hospital",  "Silver Plus", "hospital", 2200),
        ("HOSP_GOLD",        "Gold Hospital",         "Gold",        "hospital", 2900),
        ("EXT_BASIC",        "Starter Extras",        "Extras 1",    "extras",    400),
        ("EXT_MID",          "Everyday Extras",       "Extras 2",    "extras",    700),
        ("EXT_TOP",          "Top Extras",            "Extras 3",    "extras",   1100),
        ("COMB_BRONZE",      "Bronze Combined",       "Bronze",      "combined", 1900),
        ("COMB_SILVER",      "Silver Combined",       "Silver",      "combined", 2500),
        ("COMB_SILVER_PLUS", "Silver Plus Combined",  "Silver Plus", "combined", 2900),
        ("COMB_GOLD",        "Gold Combined",         "Gold",        "combined", 3800),
    ]
    # pricing periods: each starts 1 April except the first
    periods = [
        (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-03-31"), 1.000),
        (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31"), 1.052),
        (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31"), 1.108),
        (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"), 1.167),
    ]
    rows = []
    for code, name, tier, cover, prem in base:
        for start, end, mult in periods:
            rows.append({
                "product_code": code,
                "product_name": name,
                "tier": tier,
                "cover_type": cover,
                "base_premium": round(prem * mult, 2),
                "effective_from": start.strftime("%Y-%m-%d"),
                "effective_to": end.strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(rows)


PRODUCTS = build_products()
# quick lookup: product_code -> (tier, cover_type, base premium at 2023 prices)
PROD_INFO = {
    r.product_code: (r.tier, r.cover_type, r.base_premium)
    for r in PRODUCTS[PRODUCTS.effective_from == "2023-07-01"].itertuples()
}
ALL_CODES = list(PROD_INFO)


def premium_multiplier(d):
    """Cumulative rate rise applying on date d."""
    if d < pd.Timestamp("2024-04-01"):
        return 1.000
    if d < pd.Timestamp("2025-04-01"):
        return 1.052
    if d < pd.Timestamp("2026-04-01"):
        return 1.108
    return 1.167


# ============================================================= 2. MEMBERS
FIRST = ["Aisha", "Liam", "Mia", "Noah", "Ruby", "Ethan", "Chloe", "Jack",
         "Zara", "Oliver", "Grace", "Lucas", "Ivy", "Henry", "Ella", "Leo",
         "Hana", "Mason", "Priya", "Arjun", "Sofia", "Kai", "Layla", "Omar",
         "Isla", "Felix", "Nina", "Tom", "Amara", "Wei"]
LAST = ["Nguyen", "Smith", "Patel", "Wilson", "Chen", "Brown", "Singh",
        "Taylor", "Kumar", "Walker", "Ali", "Murphy", "Tran", "Clarke",
        "Rossi", "Lee", "Hall", "Osman", "Baker", "Zhang", "Novak", "Reid",
        "Haddad", "Fischer", "Kelly", "Mehta", "Young", "Dias", "Barnes", "Yilmaz"]

STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
STATE_W = [0.31, 0.26, 0.20, 0.11, 0.07, 0.02, 0.02, 0.01]        # R3
PC_START = {"NSW": 2000, "VIC": 3000, "QLD": 4000, "WA": 6000,
            "SA": 5000, "TAS": 7000, "ACT": 2600, "NT": 800}

CHANNELS = {                                                       # R2, R8
    "comparison_site": {"w": .24, "mu": 30, "sd": 8,  "churn": 1.9, "cac": 180},
    "direct_online":   {"w": .27, "mu": 37, "sd": 12, "churn": 1.1, "cac": 120},
    "broker":          {"w": .14, "mu": 45, "sd": 13, "churn": 1.6, "cac": 320},
    "retail_branch":   {"w": .13, "mu": 56, "sd": 12, "churn": 0.7, "cac": 260},
    "call_centre":     {"w": .14, "mu": 49, "sd": 14, "churn": 1.0, "cac": 210},
    "employer":        {"w": .08, "mu": 41, "sd": 10, "churn": 0.6, "cac":  90},
}
CH_NAMES = list(CHANNELS)


def build_members():
    n = N_MEMBERS
    channel = rng.choice(CH_NAMES, n, p=[CHANNELS[c]["w"] for c in CH_NAMES])

    age = np.array([rng.normal(CHANNELS[c]["mu"], CHANNELS[c]["sd"]) for c in channel])
    age = np.clip(age, 18, 92).round().astype(int)

    # join dates spread over the history window, weighted to recent
    span = (HISTORY_END - HISTORY_START).days
    join = HISTORY_START + pd.to_timedelta((rng.beta(1.6, 1.0, n) * span).astype(int), unit="D")

    # FIXED: birthday scattered across the year rather than all on one date
    dob = join - pd.to_timedelta(age * 365.25 + rng.integers(0, 365, n), unit="D")
    dob = dob.floor("D")
    age_at_join = ((join - dob).days / 365.25).astype(int)

    # R1: Lifetime Health Cover loading
    lhc = np.clip(np.where(age_at_join > 30, (age_at_join - 30) * 2, 0), 0, 70)
    lhc = np.where(rng.random(n) < 0.55, 0, lhc)   # 55% held continuous prior cover

    state = rng.choice(STATES, n, p=STATE_W)
    roll = rng.random(n)
    income_tier = np.where(roll < .58, "base",
                   np.where(roll < .80, "tier1",
                    np.where(roll < .93, "tier2", "tier3")))

    df = pd.DataFrame({
        "member_id": [f"M{100000 + i}" for i in range(n)],
        "first_name": rng.choice(FIRST, n),
        "last_name": rng.choice(LAST, n),
        "date_of_birth": dob.strftime("%Y-%m-%d"),
        "gender": rng.choice(["M", "F"], n, p=[.49, .51]),
        "state": state,
        "postcode": [PC_START[s] + rng.integers(0, 800) for s in state],
        "income_tier": income_tier,
        "lhc_loading_pct": lhc,
        "join_date": join,
        "acquisition_channel": channel,
    })
    # keep clean copies for the simulation before we dirty the output
    clean = df[["member_id", "join_date", "acquisition_channel", "state"]].copy()
    clean["age_at_join"] = age_at_join
    return df, clean


def dirty_members(df):
    n = len(df)
    # F1: Victoria three ways
    vic = df.index[df.state == "VIC"]
    mess = rng.choice(vic, int(len(vic) * .30), replace=False)
    h = len(mess) // 2
    df.loc[mess[:h], "state"] = "Vic"
    df.loc[mess[h:], "state"] = "Victoria"

    # F2: gender six ways
    g = rng.random(n)
    df.loc[(g > .70) & (df.gender == "M"), "gender"] = "Male"
    df.loc[(g > .70) & (df.gender == "F"), "gender"] = "Female"
    df.loc[g > .96, "gender"] = "U"
    df.loc[g > .99, "gender"] = ""

    # F3: join_date in three formats
    f = rng.integers(0, 3, n)
    jd = df.join_date
    df["join_date"] = np.select(
        [f == 0, f == 1, f == 2],
        [jd.dt.strftime("%Y-%m-%d"), jd.dt.strftime("%d/%m/%Y"), jd.dt.strftime("%Y%m%d")])

    # F4: 3% missing join_date
    df.loc[rng.random(n) < .03, "join_date"] = ""

    # F5: 40 near-duplicates
    src = rng.choice(n, 40, replace=False)
    d = df.loc[src].copy()
    d["first_name"] = d.first_name.str.upper() + "  "
    d["last_name"] = " " + d.last_name.str.lower()
    df = pd.concat([df, d], ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ============================================================ 3. POLICIES
SCALES = ["Single", "Couple", "Family", "Single Parent"]
SCALE_W = [.44, .26, .22, .08]
SCALE_MULT = {"Single": 1.0, "Couple": 2.0, "Family": 2.1, "Single Parent": 1.6}
EXCESS = [0, 250, 500, 750]
EXCESS_W = [.22, .34, .30, .14]
EXCESS_MULT = {0: 1.0, 250: .93, 500: .87, 750: .82}


def build_policies(members_clean):
    """Most members hold one policy, some hold two (hospital + extras)."""
    rows = []
    pid = 500000
    for m in members_clean.itertuples():
        n_pol = 2 if rng.random() < 0.12 else 1
        if n_pol == 2:
            codes = [rng.choice([c for c in ALL_CODES if PROD_INFO[c][1] == "hospital"]),
                     rng.choice([c for c in ALL_CODES if PROD_INFO[c][1] == "extras"])]
        else:
            codes = [rng.choice(ALL_CODES, p=_code_weights())]

        scale = rng.choice(SCALES, p=SCALE_W)
        for code in codes:
            excess = int(rng.choice(EXCESS, p=EXCESS_W)) if PROD_INFO[code][1] != "extras" else 0
            rows.append({
                "policy_id": f"P{pid}",
                "member_id": m.member_id,
                "product_code": code,
                "scale": scale,
                "excess": excess,
                "policy_start_date": m.join_date,
                "age_at_join": m.age_at_join,
                "channel": m.acquisition_channel,
                "state": m.state,
            })
            pid += 1
    return pd.DataFrame(rows)


def _code_weights():
    """Skew the single-policy mix towards mid tiers and combined cover."""
    w = []
    for c in ALL_CODES:
        tier, cover, _ = PROD_INFO[c]
        base = {"hospital": .9, "extras": .7, "combined": 1.5}[cover]
        if tier in ("Silver", "Bronze"):
            base *= 1.4
        if tier == "Gold":
            base *= 1.1
        w.append(base)
    w = np.array(w)
    return w / w.sum()


def annual_premium(code, scale, excess, lhc_pct, age_at_join, on_date):
    tier, cover, base = PROD_INFO[code]
    p = base * premium_multiplier(on_date) * SCALE_MULT[scale] * EXCESS_MULT[excess]
    p *= (1 + lhc_pct / 100)
    if 18 <= age_at_join <= 29:                                    # R9
        p *= 0.92
    return round(p, 2)


# ====================================================== 4. THE SIMULATION
def simulate(policies, members_clean):
    """
    Walk month by month. Each month, for every active policy:
      - maybe generate claims
      - maybe generate a service interaction or complaint
      - decide whether it lapses, downgrades, or continues
    Claims and complaints feed the lapse decision, which is why this has to
    be a loop rather than three independent random draws.
    """
    lhc_map = dict(zip(members_clean.member_id, members_clean.age_at_join))
    lhc_pct = {}
    for m in members_clean.itertuples():
        lhc_pct[m.member_id] = 0  # filled below from the members frame

    events, claims, interactions = [], [], []
    state = {}
    for p in policies.itertuples():
        state[p.policy_id] = {
            "member_id": p.member_id, "code": p.product_code, "scale": p.scale,
            "excess": p.excess, "start": p.policy_start_date, "end": None,
            "status": "active", "channel": p.channel, "state": p.state,
            "age_at_join": p.age_at_join, "claims_12m": 0,
            "open_complaint_until": None,
        }
        events.append({"policy_id": p.policy_id, "event_date": p.policy_start_date,
                       "event_type": "join", "from_product": "", "to_product": p.product_code,
                       "retention_offer_flag": False})

    months = pd.date_range(HISTORY_START, HISTORY_END, freq="MS")
    claim_id, inter_id, event_id = 900000, 700000, 800000

    for month in months:
        for pol_id, s in state.items():
            if s["status"] != "active" or s["start"] > month:
                continue

            tier, cover, _ = PROD_INFO[s["code"]]
            tenure_m = (month.year - s["start"].year) * 12 + month.month - s["start"].month
            tenure_y = tenure_m / 12

            # ---------------------------------------------------- claims
            # R11: waiting periods suppress first-year claims
            wait_factor = 0.25 if tenure_m < 12 else 1.0
            if cover in ("hospital", "combined"):
                rate = 0.020 * wait_factor * (1.5 if tier == "Gold" else 1.0)
                # R10: Gold Family in WA and SA claims harder
                if tier == "Gold" and s["scale"] == "Family" and s["state"] in ("WA", "SA"):
                    rate *= 2.2
                if s["age_at_join"] >= 60:
                    rate *= 1.8
                if 18 <= s["age_at_join"] <= 29:
                    rate *= 0.5                                     # R9
                if rng.random() < rate:
                    charged = float(np.round(rng.lognormal(8.4, 0.7), 2))
                    benefit = round(charged * rng.uniform(0.72, 0.95) - s["excess"] * 0.5, 2)
                    claims.append({
                        "claim_id": f"C{claim_id}", "policy_id": pol_id,
                        "member_id": s["member_id"], "claim_type": "hospital",
                        "service_date": month + pd.Timedelta(days=int(rng.integers(0, 28))),
                        "service_category": rng.choice(
                            ["Orthopaedic", "Cardiac", "Obstetrics", "General Surgery",
                             "Oncology", "Rehabilitation", "Endoscopy"]),
                        "charged_amount": charged, "benefit_paid": max(benefit, 0),
                        "excess_applied": s["excess"],
                    })
                    claim_id += 1

            if cover in ("extras", "combined"):
                rate = 0.30 * (0.6 if tenure_m < 12 else 1.0)
                if rng.random() < rate:
                    charged = float(np.round(rng.lognormal(4.6, 0.6), 2))
                    claims.append({
                        "claim_id": f"C{claim_id}", "policy_id": pol_id,
                        "member_id": s["member_id"], "claim_type": "extras",
                        "service_date": month + pd.Timedelta(days=int(rng.integers(0, 28))),
                        "service_category": rng.choice(
                            ["Dental", "Optical", "Physiotherapy", "Chiropractic",
                             "Psychology", "Remedial Massage", "Podiatry"]),
                        "charged_amount": charged,
                        "benefit_paid": round(charged * rng.uniform(0.55, 0.80), 2),
                        "excess_applied": 0,
                    })
                    claim_id += 1
                    s["claims_12m"] += 1

            # ---------------------------------------------- interactions
            if rng.random() < 0.09:
                is_complaint = rng.random() < 0.16
                resolved = rng.random() < 0.72 if is_complaint else True
                interactions.append({
                    "interaction_id": f"I{inter_id}", "member_id": s["member_id"],
                    "interaction_date": month + pd.Timedelta(days=int(rng.integers(0, 28))),
                    "channel": rng.choice(["phone", "web_chat", "app", "email", "branch"],
                                          p=[.38, .22, .18, .16, .06]),
                    "reason_code": rng.choice(
                        ["claim_query", "premium_query", "cover_change", "billing_issue",
                         "provider_search", "complaint", "cancellation_enquiry"]),
                    "complaint_flag": is_complaint,
                    "resolved_flag": resolved,
                    "handle_time_secs": int(rng.integers(60, 1800)),
                })
                inter_id += 1
                if is_complaint and not resolved:                   # R7
                    s["open_complaint_until"] = month + pd.Timedelta(days=90)

            # ------------------------------------------------ lapse risk
            hazard = 0.010
            hazard *= CHANNELS[s["channel"]]["churn"]               # R8
            if tenure_y < 1:
                hazard *= 1.6
            elif tenure_y > 4:
                hazard *= 1.25
            if month.month in (4, 5):                               # R4
                hazard *= 3.0 if tier == "Gold" else 2.0
                if tenure_y > 4:
                    hazard *= 1.4
            if cover == "extras" and s["claims_12m"] == 0 and tenure_m >= 12:
                hazard *= 3.0                                       # R6
            if s["open_complaint_until"] and month <= s["open_complaint_until"]:
                hazard *= 2.6                                       # R7
            if 18 <= s["age_at_join"] <= 29:
                hazard *= 1.5                                       # R9

            if rng.random() < hazard:
                # R5: a retention offer is made to some at-risk members,
                # and about a third of those downgrade instead of leaving
                offered = rng.random() < 0.35
                downgrade_path = {
                    "COMB_GOLD": "COMB_SILVER_PLUS", "HOSP_GOLD": "HOSP_SILVER_PLUS",
                    "COMB_SILVER_PLUS": "COMB_SILVER", "HOSP_SILVER_PLUS": "HOSP_SILVER",
                    "COMB_SILVER": "COMB_BRONZE", "HOSP_SILVER": "HOSP_BRONZE",
                    "EXT_TOP": "EXT_MID", "EXT_MID": "EXT_BASIC",
                }
                target = downgrade_path.get(s["code"])
                if offered and target and rng.random() < 0.55:
                    events.append({
                        "policy_id": pol_id,
                        "event_date": month + pd.Timedelta(days=int(rng.integers(0, 28))),
                        "event_type": "downgrade", "from_product": s["code"],
                        "to_product": target, "retention_offer_flag": True})
                    s["code"] = target
                else:
                    end = month + pd.Timedelta(days=int(rng.integers(0, 28)))
                    events.append({
                        "policy_id": pol_id, "event_date": end,
                        "event_type": "lapse", "from_product": s["code"],
                        "to_product": "", "retention_offer_flag": offered})
                    s["status"] = "lapsed"
                    s["end"] = end

            # rolling 12m claim counter decays
            if month.month == s["start"].month:
                s["claims_12m"] = 0

    return state, events, claims, interactions


# ================================================================== BUILD
print("building members...")
members_raw, members_clean = build_members()

print("building policies...")
policies = build_policies(members_clean)
print(f"  {len(policies):,} policies for {N_MEMBERS:,} members")

print("simulating 36 months (this takes a minute)...")
final_state, events, claims, interactions = simulate(policies, members_clean)

# ------------------------------------------------- assemble the policies file
lhc_lookup = dict(zip(members_raw.member_id, members_raw.lhc_loading_pct))
pol_rows = []
for pol_id, s in final_state.items():
    prem = annual_premium(s["code"], s["scale"], s["excess"],
                          lhc_lookup[s["member_id"]], s["age_at_join"], s["start"])
    pol_rows.append({
        "policy_id": pol_id,
        "member_id": s["member_id"],
        "product_code": s["code"],
        "scale": s["scale"],
        "excess": s["excess"],
        "annual_premium": f"${prem:,.2f}",                          # F7
        "policy_start_date": s["start"].strftime("%d/%m/%Y"),       # F3
        "policy_end_date": s["end"].strftime("%d/%m/%Y") if s["end"] else "",
        "status": s["status"],
    })
policies_out = pd.DataFrame(pol_rows)

events_out = pd.DataFrame(events)
events_out["event_id"] = [f"E{800000 + i}" for i in range(len(events_out))]
events_out["event_date"] = pd.to_datetime(events_out.event_date).dt.strftime("%Y-%m-%d")
events_out = events_out[["event_id", "policy_id", "event_date", "event_type",
                         "from_product", "to_product", "retention_offer_flag"]]

claims_out = pd.DataFrame(claims)
claims_out["service_date"] = pd.to_datetime(claims_out.service_date).dt.strftime("%Y%m%d")  # F3
# F6: claims system zero-pads the member id, policy admin does not
claims_out["member_id"] = claims_out.member_id.str.replace("M", "M0", regex=False)

inter_out = pd.DataFrame(interactions)
inter_out["interaction_date"] = pd.to_datetime(inter_out.interaction_date).dt.strftime("%Y-%m-%d")

# ------------------------------------------------------- remaining faults
# F8: ~0.8% of claims are reversals, stored as negatives
rev = rng.choice(len(claims_out), int(len(claims_out) * .008), replace=False)
claims_out.loc[rev, "benefit_paid"] = -claims_out.loc[rev, "benefit_paid"].abs()

# F9: 30 orphan claims pointing at policies that do not exist
orphans = claims_out.sample(30, random_state=SEED).copy()
orphans["claim_id"] = [f"C99{i:04d}" for i in range(30)]
orphans["policy_id"] = [f"P{999000 + i}" for i in range(30)]
claims_out = pd.concat([claims_out, orphans], ignore_index=True)

# F10: a few service dates in the future
fut = rng.choice(len(claims_out), 12, replace=False)
claims_out.loc[fut, "service_date"] = "20270115"

# F11: a few extras claims where benefit exceeds charge
ext_idx = claims_out.index[claims_out.claim_type == "extras"]
bad = rng.choice(ext_idx, 18, replace=False)
claims_out.loc[bad, "benefit_paid"] = claims_out.loc[bad, "charged_amount"] * 1.15

members_out = dirty_members(members_raw)

# ================================================================== WRITE
files = {
    "pas_members.csv": members_out,
    "pas_policies.csv": policies_out,
    "pas_policy_events.csv": events_out,
    "clm_claims.csv": claims_out.sample(frac=1, random_state=SEED).reset_index(drop=True),
    "cx_interactions.csv": inter_out,
    "ref_products.csv": PRODUCTS,
}
print("\nwriting files:")
for name, frame in files.items():
    path = os.path.join(OUTDIR, name)
    frame.to_csv(path, index=False)
    print(f"  {name:<26} {len(frame):>8,} rows")

lapsed = (policies_out.status == "lapsed").sum()
print(f"\nsanity: {lapsed:,} of {len(policies_out):,} policies lapsed "
      f"({lapsed / len(policies_out):.1%})")