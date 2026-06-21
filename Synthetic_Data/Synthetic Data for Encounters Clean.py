#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from scipy.stats import nbinom, lognorm, truncnorm
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════════
# SECTION 1 — PARAMETERS
# Update N_PATIENTS and OUT_PATH before running.
# ═════════════════════════════════════════════════════════════════

N_PATIENTS      = 10000   # ← total patients including edge cases
N_EDGE_PATIENTS = max(10, N_PATIENTS // 100)  # ~1% edge cases, min 10
N_MAIN_PATIENTS = N_PATIENTS - N_EDGE_PATIENTS

OUT_PATH = "/uufs/chpc.utah.edu/common/PE/proj_synthetic_EHR/Monika_Workspace/00_Data/00_Raw_Data/Encounter_full.parquet"

OBS_START = pd.Timestamp('2016-01-01')
OBS_END   = pd.Timestamp('2024-12-31')
OBS_DAYS  = (OBS_END - OBS_START).days

np.random.seed(42)

# ── Encounter count distribution (negative binomial parameters) ──
# Source: Cosmos data summary statistics
ENC_MEAN = 7.71
ENC_SD   = 29.60

# ── Inter-encounter gap (log-normal median in days) ──
GAP_MEDIAN = 7.0

# ── Encounter type probabilities ──
P_OUTPAT_ONLY = 0.491
P_ED_ONLY     = 0.060
P_BOTH        = 0.267
P_NEITHER     = 0.181  # inpatient / other

# ── Diagnosis code counts per encounter (negative binomial parameters) ──
OUTPAT_PRI_MEAN  = 2.39;  OUTPAT_PRI_SD  = 2.53;  OUTPAT_PRI_ZERO  = 0.242
OUTPAT_NONPRI_MEAN = 6.05; OUTPAT_NONPRI_SD = 6.77; OUTPAT_NONPRI_ZERO = 0.217
ED_PRI_MEAN      = 0.90;  ED_PRI_SD      = 1.95;  ED_PRI_ZERO      = 0.673
NEITHER_NONPRI_MEAN = 2.86; NEITHER_NONPRI_ZERO = 0.769

# ═════════════════════════════════════════════════════════════════
# SECTION 2 — TOP 10 DIAGNOSIS CODE FREQUENCIES
# Observed frequencies from Cosmos data.
# Remaining ~30% of probability mass is distributed across all
# ICD-10-CM codes using a Zipf (rank-inverse) distribution.
# ═════════════════════════════════════════════════════════════════

top10_outpat_primary = {
    'F33.3': 0.2064, 'F41.1': 0.0823, 'F84.0': 0.0793,
    'F32.3': 0.0730, 'F31.2': 0.0634, 'F33.2': 0.0604,
    'F31.5': 0.0482, 'F43.10': 0.0454, 'F32.4': 0.0449,
    'F33.1': 0.0419
}
top10_outpat_nonprimary = {
    'F84.0': 0.2611, 'F33.3': 0.2530, 'F41.1': 0.2121,
    'F41.9': 0.1510, 'F43.10': 0.1216, 'Z79.899': 0.0949,
    'F90.2': 0.0941, 'F32.3': 0.0926, 'F32.4': 0.0841,
    'F33.2': 0.0766
}
top10_ed_primary = {
    'R45.851': 0.0763, 'F32.A': 0.0374, 'F33.3': 0.0345,
    'F33.2': 0.0215, 'F31.9': 0.0210, 'F32.9': 0.0201,
    'F41.9': 0.0178, 'F29': 0.0165, 'F31.2': 0.0164,
    'R46.89': 0.0114
}
# Inpatient/other reuses outpatient non-primary as a proxy
top10_neither_nonprimary = top10_outpat_nonprimary.copy()

# ═════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD ICD-10-CM CODE LIST
# Attempts to download the full 2024 code list from the CDC.
# Falls back to top-10 codes only if the download fails.
# ═════════════════════════════════════════════════════════════════

def add_dot(code):
    """Convert CDC format (A000) to standard format (A00.0)."""
    code = code.strip()
    if '.' in code or len(code) <= 3:
        return code
    return code[:3] + '.' + code[3:]

print("Loading ICD-10-CM codes...")
try:
    import urllib.request, zipfile, io
    url = ("https://ftp.cdc.gov/pub/health_statistics/nchs/publications/"
           "ICD10CM/2024/icd10cm-CodesDescriptions-2024.zip")
    zf       = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
    with zf.open('icd10cm-codes-2024.txt') as f:
        all_codes = [add_dot(ln.split(' ')[0]) for ln in f.read().decode().splitlines() if ln.strip()]
    print(f"  Loaded {len(all_codes):,} ICD-10-CM codes from CDC.")
except Exception as e:
    print(f"  CDC download failed ({e}). Using top-10 codes as fallback.")
    all_codes = list({
        *top10_outpat_primary, *top10_outpat_nonprimary, *top10_ed_primary
    })

# ═════════════════════════════════════════════════════════════════
# SECTION 4 — HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def build_sampling_weights(top10_dict, all_codes):
    """
    Combine known top-10 frequencies with a Zipf tail for all other codes.
    The top-10 codes account for ~70% of encounters; the remaining ~30%
    is distributed across all other ICD-10 codes with frequency ∝ 1/rank.
    """
    top10_codes  = list(top10_dict.keys())
    top10_freqs  = list(top10_dict.values())
    remaining    = max(0.0, 1.0 - sum(top10_freqs))
    other_codes  = [c for c in all_codes if c not in top10_codes]
    tail_ranks   = np.arange(11, 11 + len(other_codes), dtype=float)
    zipf_weights = (1.0 / tail_ranks)
    zipf_weights = (zipf_weights / zipf_weights.sum()) * remaining
    weights      = np.array(top10_freqs + zipf_weights.tolist())
    return top10_codes + other_codes, weights / weights.sum()

def fit_nbinom(mean, sd):
    """
    Moment-match a negative binomial to observed mean and SD.
    Returns (n, p) parameters for scipy.stats.nbinom.
    The negative binomial is used instead of Poisson because clinical
    utilization data is overdispersed (variance >> mean).
    """
    var = max(sd ** 2, mean + 0.01)
    p   = mean / var
    n   = mean * p / (1 - p)
    return n, p

def draw_nbinom(n, p, zero_pct=0.0, min_val=0):
    """
    Draw one sample from a zero-inflated negative binomial.
    zero_pct handles structural zeros (fields never filled in),
    separate from distributional zeros.
    """
    if zero_pct > 0 and np.random.random() < zero_pct:
        return 0
    return max(int(nbinom.rvs(n, p)), min_val)

def sample_codes(n_codes, codes, weights):
    """
    Sample n_codes unique ICD-10 codes without replacement,
    weighted by the combined top-10 + Zipf distribution.
    """
    if n_codes == 0:
        return []
    mask       = weights > 0
    codes_nz   = np.array(codes)[mask]
    weights_nz = weights[mask] / weights[mask].sum()
    return np.random.choice(
        codes_nz, size=min(n_codes, len(codes_nz)),
        replace=False, p=weights_nz
    ).tolist()

def generate_encounter_dates(index_date, n_encounters):
    """
    Space encounters through time using a log-normal gap distribution.
    Log-normal is appropriate because inter-event times in healthcare
    are right-skewed and strictly positive.
    Encounters beyond OBS_END are silently dropped.
    """
    if n_encounters == 1:
        return [index_date]
    mu_log, sigma_log = np.log(max(GAP_MEDIAN, 1)), 0.8
    dates = [index_date]
    for _ in range(n_encounters - 1):
        gap       = max(1, int(lognorm.rvs(s=sigma_log, scale=np.exp(mu_log))))
        next_date = dates[-1] + pd.Timedelta(days=gap)
        if next_date > OBS_END:
            break
        dates.append(next_date)
    return dates

def generate_continuous(n, mean, sd, low, high, missing_rate, is_count=False):
    """Draw from a truncated normal; optionally round to integer counts."""
    if missing_rate >= 1.0:
        return np.full(n, np.nan)
    a, b   = (low - mean) / sd, (high - mean) / sd
    values = truncnorm.rvs(a, b, loc=mean, scale=sd, size=n).astype(float)
    if is_count:
        values = np.clip(np.round(values), low, high)
    if missing_rate > 0:
        values[np.random.random(n) < missing_rate] = np.nan
    return values

def generate_skewed_years(n, obs_start, obs_end, missing_rate):
    """Generate years exponentially skewed toward obs_end."""
    years   = np.arange(obs_start.year, obs_end.year + 1)
    weights = np.exp(np.linspace(0, 3, len(years)))
    weights /= weights.sum()
    drawn        = np.random.choice(years, size=n, p=weights)
    missing_mask = np.random.random(n) < missing_rate
    return [None if missing_mask[i] else int(drawn[i]) for i in range(n)]

# ── Pre-fit all negative binomial distributions ──
n_enc,            p_enc            = fit_nbinom(ENC_MEAN,             ENC_SD)
n_outpat_pri,     p_outpat_pri     = fit_nbinom(OUTPAT_PRI_MEAN,      OUTPAT_PRI_SD)
n_outpat_nonpri,  p_outpat_nonpri  = fit_nbinom(OUTPAT_NONPRI_MEAN,   OUTPAT_NONPRI_SD)
n_ed_pri,         p_ed_pri         = fit_nbinom(ED_PRI_MEAN,          ED_PRI_SD)
n_neither_nonpri, p_neither_nonpri = fit_nbinom(NEITHER_NONPRI_MEAN,  1.5)

# ── Pre-build ICD-10 sampling distributions ──
outpat_pri_codes,     outpat_pri_weights     = build_sampling_weights(top10_outpat_primary,     all_codes)
outpat_nonpri_codes,  outpat_nonpri_weights  = build_sampling_weights(top10_outpat_nonprimary,  all_codes)
ed_pri_codes,         ed_pri_weights         = build_sampling_weights(top10_ed_primary,         all_codes)
neither_nonpri_codes, neither_nonpri_weights = build_sampling_weights(top10_neither_nonprimary, all_codes)

ENC_TYPES  = ['outpat_only', 'ed_only', 'both', 'neither']
ENC_TYPE_P = np.array([P_OUTPAT_ONLY, P_ED_ONLY, P_BOTH, P_NEITHER])
ENC_TYPE_P = ENC_TYPE_P / ENC_TYPE_P.sum()

# ═════════════════════════════════════════════════════════════════
# SECTION 5 — GENERATE ENCOUNTERS
# For each patient: draw encounter count → assign random index date
# → space encounters using log-normal gaps → assign encounter type
# → sample diagnosis codes appropriate to that type.
# ═════════════════════════════════════════════════════════════════

print(f"Generating encounters for {N_PATIENTS} patients...")
records = []

for patient_id in range(N_MAIN_PATIENTS):
    if patient_id % 1000 == 0 and patient_id > 0:
        print(f"  {patient_id:,} patients done...")

    n_enc_patient = max(1, draw_nbinom(n_enc, p_enc))
    index_date    = OBS_START + pd.Timedelta(days=int(np.random.randint(0, OBS_DAYS)))
    enc_dates     = generate_encounter_dates(index_date, n_enc_patient)

    for enc_num, enc_date in enumerate(enc_dates):
        enc_id   = f"P{patient_id}_E{enc_num}"
        enc_type = np.random.choice(ENC_TYPES, p=ENC_TYPE_P)

        if enc_type == 'outpat_only':
            n_pri = draw_nbinom(n_outpat_pri, p_outpat_pri, OUTPAT_PRI_ZERO,  min_val=1)
            n_non = draw_nbinom(n_outpat_nonpri, p_outpat_nonpri, OUTPAT_NONPRI_ZERO)
            records.append({
                'PatientDurableKey':  patient_id,
                'EncounterKey':       enc_id,
                'EncDate':            enc_date,
                'EncounterType':      'outpatient',
                'OutpatPrimaryDx':    ', '.join(sample_codes(n_pri, outpat_pri_codes,    outpat_pri_weights))    or None,
                'OutpatNonPrimaryDx': ', '.join(sample_codes(n_non, outpat_nonpri_codes, outpat_nonpri_weights)) or None,
                'EDPrimDx':           None,
            })

        elif enc_type == 'ed_only':
            n_ed = draw_nbinom(n_ed_pri, p_ed_pri, ED_PRI_ZERO, min_val=1)
            records.append({
                'PatientDurableKey':  patient_id,
                'EncounterKey':       enc_id,
                'EncDate':            enc_date,
                'EncounterType':      'ed',
                'OutpatPrimaryDx':    None,
                'OutpatNonPrimaryDx': None,
                'EDPrimDx':           ', '.join(sample_codes(n_ed, ed_pri_codes, ed_pri_weights)) or None,
            })

        elif enc_type == 'both':
            n_pri = draw_nbinom(n_outpat_pri,   p_outpat_pri,   OUTPAT_PRI_ZERO,  min_val=1)
            n_non = draw_nbinom(n_outpat_nonpri, p_outpat_nonpri, OUTPAT_NONPRI_ZERO)
            n_ed  = draw_nbinom(n_ed_pri,        p_ed_pri,        ED_PRI_ZERO,     min_val=1)
            records.append({
                'PatientDurableKey':  patient_id,
                'EncounterKey':       enc_id,
                'EncDate':            enc_date,
                'EncounterType':      'both',
                'OutpatPrimaryDx':    ', '.join(sample_codes(n_pri, outpat_pri_codes,    outpat_pri_weights))    or None,
                'OutpatNonPrimaryDx': ', '.join(sample_codes(n_non, outpat_nonpri_codes, outpat_nonpri_weights)) or None,
                'EDPrimDx':           ', '.join(sample_codes(n_ed,  ed_pri_codes,        ed_pri_weights))        or None,
            })

        else:  # neither — inpatient / other
            n_non = draw_nbinom(n_neither_nonpri, p_neither_nonpri, NEITHER_NONPRI_ZERO)
            records.append({
                'PatientDurableKey':  patient_id,
                'EncounterKey':       enc_id,
                'EncDate':            enc_date,
                'EncounterType':      'inpatient_other',
                'OutpatPrimaryDx':    None,
                'OutpatNonPrimaryDx': ', '.join(sample_codes(n_non, neither_nonpri_codes, neither_nonpri_weights)) or None,
                'EDPrimDx':           None,
            })

enc_df = pd.DataFrame(records).sort_values(['PatientDurableKey', 'EncDate']).reset_index(drop=True)
n_enc_rows = len(enc_df)
print(f"  Generated {n_enc_rows:,} encounters.")

# ═════════════════════════════════════════════════════════════════
# SECTION 6 — BINARY CLINICAL VARIABLES
# Y/N flags for medications, diagnoses, and insurance status.
# Prevalences informed by Cosmos data summary statistics.
# ═════════════════════════════════════════════════════════════════

binary_vars = {
    # Anthropometrics
    'WeightGain10_20lbs': 0.08, 'WeightGain20pluslbs': 0.05,
    'WeightLoss10_20lbs': 0.07, 'WeightLoss20pluslbs': 0.04,
    # Insurance
    'HealthInsLoss': 0.06, 'MedicareYN': 0.18, 'MedicaidYN': 0.22,
    'SelfPayYN': 0.12, 'MiscOtherYN': 0.05,
    # Suicide
    'HxSuicideAttempt': 0.18, 'HxSuicideAttempt30DaysPrior': 0.06,
    'CurrentSuicideAttempt': 0.04,
    # Medications
    'SSRI': 0.35, 'Antipsychotics': 0.28, 'Lithium': 0.12,
    'Anticonvulsants': 0.15, 'Benzos': 0.14, 'NRI': 0.08,
    'ADHDStimulants': 0.16, 'PrescripMed45': 0.20, 'PrescripMedGT5': 0.15,
    'Opioids': 0.06,
    # Diagnoses
    'Depression': 0.45, 'DepressionPlus': 0.22, 'MDDActive': 0.30,
    'MDDRemission': 0.18, 'AnxietyPlus': 0.25, 'Anxiety': 0.35,
    'AdjStressDisorder': 0.15, 'MoodDisorders': 0.40, 'PsychoticPlus': 0.12,
    'Schizophrenia': 0.08, 'PersDisorders': 0.10, 'MentalDisorder': 0.50,
    'EatingDisorder': 0.06, 'SexualDysfunction': 0.07, 'SleepDisorder': 0.18,
    'Autism': 0.12, 'PTSD': 0.15, 'PTSDPlus': 0.10,
    'ImpulseDisorders': 0.08, 'ExternalizingDis': 0.10, 'ADHD': 0.18,
    'MalignantNeoplasm': 0.04,
}

for col, p_yes in binary_vars.items():
    enc_df[col] = np.where(np.random.binomial(1, p_yes, n_enc_rows) == 1, 'Y', 'N')

# ═════════════════════════════════════════════════════════════════
# SECTION 7 — ORDINAL RISK SCORE LEVELS & DATEKEYS
# Seven risk scores, each with a level (Low/Medium/High) and a date.
# Score and DateKey are grouped together for each score.
# ═════════════════════════════════════════════════════════════════

RISK_LEVELS = ['Low Risk', 'Medium Risk', 'Medium High Risk', 'High Risk']

score_configs = {
    'Score5':  {'level_probs': [0.40, 0.24, 0.20, 0.16], 'score_mean': 2.56,  'score_sd': 1.03,  'score_low': 0,    'score_high': 3,    'score_missing': 1.0,  'date_missing': 0.25},
    'Score49': {'level_probs': [0.16, 0.20, 0.24, 0.40], 'score_mean': 1.74,  'score_sd': 1.62,  'score_low': 0,    'score_high': 9,    'score_missing': 0.786,'date_missing': 0.30},
    'Score50': {'level_probs': [0.40, 0.24, 0.22, 0.14], 'score_mean': 5.62,  'score_sd': 7.09,  'score_low': 0.10, 'score_high': 58.10,'score_missing': 0.911,'date_missing': 0.25},
    'Score51': {'level_probs': [0.14, 0.22, 0.24, 0.40], 'score_mean': 37.14, 'score_sd': 25.57, 'score_low': 1.40, 'score_high': 98.50,'score_missing': 0.949,'date_missing': 0.30},
    'Score52': {'level_probs': [0.40, 0.26, 0.20, 0.14], 'score_mean': 40.28, 'score_sd': 26.73, 'score_low': 1,    'score_high': 99,   'score_missing': 0.496,'date_missing': 0.28},
    'Score53': {'level_probs': [0.16, 0.22, 0.22, 0.40], 'score_mean': 3.62,  'score_sd': 1.08,  'score_low': 0,    'score_high': 5,    'score_missing': 0.924,'date_missing': 0.28},
    'Score54': {'level_probs': [0.40, 0.24, 0.20, 0.16], 'score_mean': 5.90,  'score_sd': 7.12,  'score_low': 0.20, 'score_high': 83.40,'score_missing': 0.917,'date_missing': 0.30},
}

for score_name, cfg in score_configs.items():
    probs = np.array(cfg['level_probs']); probs /= probs.sum()
    enc_df[f'{score_name}Score'] = generate_continuous(
        n_enc_rows, cfg['score_mean'], cfg['score_sd'],
        cfg['score_low'], cfg['score_high'], cfg['score_missing'])
    enc_df[f'{score_name}Level']   = np.random.choice(RISK_LEVELS, size=n_enc_rows, p=probs)
    enc_df[f'{score_name}DateKey'] = generate_skewed_years(
        n_enc_rows, OBS_START, OBS_END, cfg['date_missing'])

# ═════════════════════════════════════════════════════════════════
# SECTION 8 — DEMOGRAPHICS & ANTHROPOMETRICS
# Age, height, weight, and BMI are generated at the patient level
# then merged back to encounters so they vary realistically over time.
# ═════════════════════════════════════════════════════════════════

print("Generating demographics and anthropometrics...")

patient_spine = (
    enc_df.groupby('PatientDurableKey')['EncDate']
    .min().reset_index()
    .rename(columns={'EncDate': 'first_enc'})
)
n_pat = len(patient_spine)

patient_spine['age_at_first_enc'] = truncnorm.rvs(
    (5-35)/15, (85-35)/15, loc=35, scale=15, size=n_pat).astype(int)
patient_spine['is_female']  = np.random.binomial(1, 0.55, n_pat).astype(bool)
patient_spine['adult_height'] = np.where(
    patient_spine['is_female'],
    truncnorm.rvs((56-64)/3, (72-64)/3, loc=64, scale=3, size=n_pat),
    truncnorm.rvs((61-69)/3, (78-69)/3, loc=69, scale=3, size=n_pat))
patient_spine['baseline_bmi']       = truncnorm.rvs((16-29)/6, (55-29)/6, loc=29, scale=6, size=n_pat)
patient_spine['bmi_drift_per_year'] = truncnorm.rvs((-3-0.3)/0.8, (3-0.3)/0.8, loc=0.3, scale=0.8, size=n_pat)

enc_df = enc_df.merge(patient_spine, on='PatientDurableKey', how='left')
enc_df['years_elapsed'] = (enc_df['EncDate'] - enc_df['first_enc']).dt.days / 365.25
enc_df['AgeInYears']    = (enc_df['age_at_first_enc'] + enc_df['years_elapsed']).round(1)

def compute_height(row):
    current_age = row['age_at_first_enc'] + row['years_elapsed']
    if row['age_at_first_enc'] >= 18:
        return round(row['adult_height'] + np.random.normal(0, 0.3), 1)
    pct_grown = max(0.60, min(1.0, 0.60 + (current_age - 5) * (0.40 / 13)))
    return round(row['adult_height'] * pct_grown + np.random.normal(0, 0.3), 1)

enc_df['EncounterHeightInches'] = enc_df.apply(compute_height, axis=1)
enc_df['EncounterBMI']          = (
    enc_df['baseline_bmi'] + enc_df['bmi_drift_per_year'] * enc_df['years_elapsed']
    + np.random.normal(0, 1.5, n_enc_rows)
).clip(14, 60).round(1)
enc_df['EncounterWeightLbs'] = (
    enc_df['EncounterBMI'] * enc_df['EncounterHeightInches'] ** 2 / 703
).round(1)

enc_df = enc_df.drop(columns=['first_enc', 'age_at_first_enc', 'is_female',
                               'adult_height', 'baseline_bmi',
                               'bmi_drift_per_year', 'years_elapsed'])

# ═════════════════════════════════════════════════════════════════
# SECTION 9 — CONTINUOUS CLINICAL VARIABLES
# Visit counts, PROMIS scores, and lab year variables.
# ═════════════════════════════════════════════════════════════════

enc_df['PsychiatryVisitCount'] = generate_continuous(n_enc_rows, 2.07, 5.31,  0, 123, 0.0,  is_count=True)
enc_df['PsychologyVisitCount'] = generate_continuous(n_enc_rows, 1.0,  4.71,  0, 91,  0.0,  is_count=True)
enc_df['EDVisitCount']         = generate_continuous(n_enc_rows, 0.71, 1.83,  0, 75,  0.0,  is_count=True)
enc_df['PxCount']              = generate_continuous(n_enc_rows, 4.74, 10.74, 0, 435, 0.0,  is_count=True)

# PROMIS pain/fatigue scores — no data collected in this population
enc_df['PainIntensityScore']   = np.nan
enc_df['PainIntensityDateKey'] = np.nan
enc_df['VASScore']             = np.nan
enc_df['VASDateKey']           = np.nan
enc_df['FatigueScore']         = np.nan
enc_df['FatigueDateKey']       = np.nan
enc_df['PainInterfScore']      = np.nan
enc_df['PainInterfDateKey']    = np.nan

# Lab test — most recent year prior to encounter
lab_missing_rates = {
    'Pulmonary': 0.60, 'FastingGlucose': 0.30, 'SpotGlucose': 0.35,
    'LipidLevels': 0.30, 'CRP': 0.55, 'Cortisol': 0.65, 'Thyroid': 0.25,
}
for col, missing_rate in lab_missing_rates.items():
    enc_df[col] = generate_skewed_years(n_enc_rows, OBS_START, OBS_END, missing_rate)

# ═════════════════════════════════════════════════════════════════
# SECTION 9.5 — GAUSSIAN COPULA (SYNTHETIC CORRELATIONS)
# Imposes a block correlation structure across all clinical variables.
# Correlations are synthetic and for pipeline testing only.
# ═════════════════════════════════════════════════════════════════

from scipy.stats import norm
from sklearn.covariance import LedoitWolf

# ── Define variable blocks ────────────────────────────────────────
copula_blocks = {
    'anthropometric': ['AgeInYears', 'EncounterWeightLbs', 'EncounterHeightInches',
                       'EncounterBMI', 'WeightGain10_20lbs', 'WeightGain20pluslbs',
                       'WeightLoss10_20lbs', 'WeightLoss20pluslbs'],
    'insurance':      ['HealthInsLoss', 'MedicareYN', 'MedicaidYN',
                       'SelfPayYN', 'MiscOtherYN'],
    'utilization':    ['PsychiatryVisitCount', 'PsychologyVisitCount',
                       'EDVisitCount', 'PxCount'],
    'suicide':        ['HxSuicideAttempt', 'HxSuicideAttempt30DaysPrior',
                       'CurrentSuicideAttempt'],
    'medications':    ['Opioids', 'SSRI', 'Antipsychotics', 'Lithium',
                       'Anticonvulsants', 'Benzos', 'NRI', 'ADHDStimulants',
                       'PrescripMed45', 'PrescripMedGT5'],
    'mood':           ['Depression', 'DepressionPlus', 'MDDActive', 'MDDRemission',
                       'MoodDisorders'],
    'anxiety':        ['Anxiety', 'AnxietyPlus', 'AdjStressDisorder', 'PTSD', 'PTSDPlus'],
    'psychotic':      ['PsychoticPlus', 'Schizophrenia'],
    'neurodevelop':   ['Autism', 'ADHD', 'ImpulseDisorders', 'ExternalizingDis'],
    'other_dx':       ['PersDisorders', 'MentalDisorder', 'EatingDisorder',
                       'SexualDysfunction', 'SleepDisorder', 'MalignantNeoplasm'],
    'scores':         ['Score5Score', 'Score49Score', 'Score50Score', 'Score51Score',
                       'Score52Score', 'Score53Score', 'Score54Score'],
    'labs':           ['FastingGlucose', 'SpotGlucose', 'LipidLevels',
                       'CRP', 'Cortisol', 'Thyroid'],
}

WITHIN_BLOCK_CORR  = 0.8   # was 0.5
BETWEEN_BLOCK_CORR = 0.15  # was 0.1

# ── Flatten to ordered variable list ─────────────────────────────
copula_vars = [v for block in copula_blocks.values() for v in block]
n_vars = len(copula_vars)

# ── Build correlation matrix ──────────────────────────────────────
R = np.full((n_vars, n_vars), BETWEEN_BLOCK_CORR)
np.fill_diagonal(R, 1.0)

# Set within-block correlations
idx_map = {v: i for i, v in enumerate(copula_vars)}
for block_vars in copula_blocks.values():
    for v1 in block_vars:
        for v2 in block_vars:
            if v1 != v2:
                R[idx_map[v1], idx_map[v2]] = WITHIN_BLOCK_CORR

# ── Ensure positive definiteness ─────────────────────────────────
# Hand-specified matrices can be non-PD; clip negative eigenvalues
eigvals, eigvecs = np.linalg.eigh(R)
eigvals = np.clip(eigvals, 1e-6, None)
R = eigvecs @ np.diag(eigvals) @ eigvecs.T
# Rescale diagonal back to 1
d = np.sqrt(np.diag(R))
R = R / np.outer(d, d)

# ── Transform each variable to uniform via empirical CDF ──────────
def to_uniform(x):
    """Empirical CDF — handles both continuous and binary columns."""
    n = len(x)
    return pd.Series(x).rank(method='average').values / (n + 1)

U = np.column_stack([to_uniform(enc_df[v].values) for v in copula_vars])

# ── Transform uniform → standard normal ──────────────────────────
Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))

# ── Sample from Gaussian copula ───────────────────────────────────
L   = np.linalg.cholesky(R)
Z_new = (L @ np.random.standard_normal((n_vars, n_enc_rows))).T
U_new = norm.cdf(Z_new)   # back to uniform

# ── Back-transform through empirical quantile function ───────────
def from_uniform(u_new, x_orig):
    """Map new uniform values back to original value distribution."""
    sorted_vals = np.sort(x_orig)
    quantiles   = np.linspace(0, 1, len(sorted_vals))
    return np.interp(u_new, quantiles, sorted_vals)

# Continuous columns — direct quantile interpolation
continuous_copula = ['AgeInYears', 'EncounterWeightLbs', 'EncounterHeightInches',
                     'EncounterBMI', 'PsychiatryVisitCount', 'PsychologyVisitCount',
                     'EDVisitCount', 'PxCount', 'Score5Score', 'Score49Score',
                     'Score50Score', 'Score51Score', 'Score52Score', 'Score53Score',
                     'Score54Score', 'FastingGlucose', 'SpotGlucose', 'LipidLevels',
                     'CRP', 'Cortisol', 'Thyroid']

# Binary columns — threshold at original prevalence
binary_copula = [v for v in copula_vars if v not in continuous_copula]

for i, var in enumerate(copula_vars):
    if var in continuous_copula:
        enc_df[var] = from_uniform(U_new[:, i], enc_df[var].values).round(1)
    else:
        # Back-transform binary: threshold at 1 - prevalence
        prevalence  = (enc_df[var] == 'Y').mean()
        enc_df[var] = np.where(U_new[:, i] > (1 - prevalence), 'Y', 'N')

print(f"Copula applied across {n_vars} variables in {len(copula_blocks)} blocks.")

# ═════════════════════════════════════════════════════════════════
# SECTION 9.6 — SYNTHETIC MISSINGNESS
# Applies three missingness mechanisms to the ~50 columns that
# currently have no missing data.
#
# Mechanism assignment (clinical rationale):
#   MCAR  — visit counts, remaining binary flags (random documentation gaps)
#   MAR   — medication and diagnosis flags (missingness depends on age
#            and encounter type — sparser in ED/inpatient settings and
#            for older patients with fragmented records)
#   Block — insurance fields, suicide flags, demographics (fields that
#            tend to be collected or omitted together within an encounter)
#
# Missing rates drawn once from Uniform(0.05, 0.30) with a fixed seed
# so the rate assignment is reproducible across runs.
# ═════════════════════════════════════════════════════════════════

MISSINGNESS_SEED = 99   # separate seed so main np.random stream is unaffected
rng_miss = np.random.default_rng(MISSINGNESS_SEED)

# ── Column group definitions ──────────────────────────────────────

MISS_BLOCK_INSURANCE = [
    'HealthInsLoss', 'MedicareYN', 'MedicaidYN', 'SelfPayYN', 'MiscOtherYN',
]

MISS_BLOCK_SUICIDE = [
    'HxSuicideAttempt', 'HxSuicideAttempt30DaysPrior', 'CurrentSuicideAttempt',
]

MISS_BLOCK_DEMOGRAPHICS = [
    'AgeInYears', 'EncounterHeightInches', 'EncounterWeightLbs', 'EncounterBMI',
]

MISS_MAR_MEDICATIONS = [
    'SSRI', 'Antipsychotics', 'Lithium', 'Anticonvulsants', 'Benzos',
    'NRI', 'ADHDStimulants', 'PrescripMed45', 'PrescripMedGT5', 'Opioids',
]

MISS_MAR_DIAGNOSES = [
    'Depression', 'DepressionPlus', 'MDDActive', 'MDDRemission',
    'AnxietyPlus', 'Anxiety', 'AdjStressDisorder', 'MoodDisorders',
    'PsychoticPlus', 'Schizophrenia', 'PersDisorders', 'MentalDisorder',
    'EatingDisorder', 'SexualDysfunction', 'SleepDisorder', 'Autism',
    'PTSD', 'PTSDPlus', 'ImpulseDisorders', 'ExternalizingDis', 'ADHD',
    'MalignantNeoplasm',
]

MISS_MAR_WEIGHT_FLAGS = [
    'WeightGain10_20lbs', 'WeightGain20pluslbs',
    'WeightLoss10_20lbs', 'WeightLoss20pluslbs',
]

MISS_MCAR = [
    'PsychiatryVisitCount', 'PsychologyVisitCount', 'EDVisitCount', 'PxCount',
]

# ── Draw base missing rates from Beta(1, 3), floor at 0.05 ───────
# Beta(1, 3) is right-skewed over [0, 1]:
#   median ~0.21, mean ~0.25, but long right tail allows rates up to ~0.95.
# Most columns will land in the 0.05–0.50 range; a meaningful minority
# will exceed 0.60, reflecting the sparse documentation common in EHR data.
# Rates drawn below 0.05 are clipped up to ensure no column is near-complete.
# One rate per column; stored in a dict for transparency/logging.

all_miss_cols = (
    MISS_BLOCK_INSURANCE + MISS_BLOCK_SUICIDE + MISS_BLOCK_DEMOGRAPHICS +
    MISS_MAR_MEDICATIONS + MISS_MAR_DIAGNOSES + MISS_MAR_WEIGHT_FLAGS +
    MISS_MCAR
)
base_rates = {
    col: float(np.clip(rng_miss.beta(1, 3), 0.05, 1.0))
    for col in all_miss_cols
}

print("\n── Section 9.6: Synthetic Missingness ───────────────────────")
print(f"  Columns targeted: {len(all_miss_cols)}")
print(f"  Base rate range:  {min(base_rates.values()):.3f} – {max(base_rates.values()):.3f}  "
      f"(median {np.median(list(base_rates.values())):.3f}, drawn from Beta(1,3) floor 0.05)")

n_rows = len(enc_df)

# ─────────────────────────────────────────────────────────────────
# MECHANISM 1 — MCAR
# Each row is independently masked with probability = base_rate.
# Applied to visit count columns; no structure beyond the rate.
# ─────────────────────────────────────────────────────────────────

for col in MISS_MCAR:
    mask = rng_miss.random(n_rows) < base_rates[col]
    enc_df.loc[mask, col] = np.nan

print(f"  MCAR applied to {len(MISS_MCAR)} columns.")

# ─────────────────────────────────────────────────────────────────
# MECHANISM 2 — MAR
# Missingness probability is a logistic function of:
#   - AgeInYears (standardised): older → higher P(missing)
#   - EncounterType: ED and inpatient encounters → higher P(missing)
#     (documentation is sparser in acute/non-outpatient settings)
#
# For column c:
#   logit(p_miss) = logit(base_rate[c])
#                  + beta_age  * age_std
#                  + beta_type * is_acute
#
# beta_age and beta_type are fixed at mild-to-moderate effect sizes
# so that the overall missing rate stays close to the drawn base rate.
# ─────────────────────────────────────────────────────────────────

BETA_AGE  = 0.4   # log-odds increase per SD of age
BETA_TYPE = 0.6   # log-odds increase for ED / inpatient encounters

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Pre-compute predictors once
age_std  = (enc_df['AgeInYears'] - enc_df['AgeInYears'].mean()) / enc_df['AgeInYears'].std()
is_acute = enc_df['EncounterType'].isin(['ed', 'inpatient_other']).astype(float)

mar_cols = MISS_MAR_MEDICATIONS + MISS_MAR_DIAGNOSES + MISS_MAR_WEIGHT_FLAGS

for col in mar_cols:
    base_logit = logit(base_rates[col])
    p_miss     = sigmoid(base_logit + BETA_AGE * age_std + BETA_TYPE * is_acute)
    mask       = rng_miss.random(n_rows) < p_miss.values
    # Binary Y/N columns → NaN represented as None (consistent with existing pattern)
    if enc_df[col].dtype == object:
        enc_df[col] = enc_df[col].astype(object)
        enc_df.loc[mask, col] = None
    else:
        enc_df.loc[mask, col] = np.nan

print(f"  MAR  applied to {len(mar_cols)} columns  "
      f"(beta_age={BETA_AGE}, beta_type={BETA_TYPE}).")

# ─────────────────────────────────────────────────────────────────
# MECHANISM 3 — BLOCK-CORRELATED MISSINGNESS
# Within each block a shared latent Bernoulli variable determines
# whether a patient-encounter's entire block tends to be missing.
#
# For row i and block B:
#   1. Draw a block-level "missingness propensity" u_i ~ Uniform(0,1)
#   2. For each column c in B, miss if:
#        u_i < base_rate[c]  OR  (independent draw < base_rate[c] * alpha)
#      where alpha controls within-block correlation strength.
#      alpha=0 → purely independent (MCAR); alpha=1 → fully driven by block propensity.
#
# alpha=0.7 means ~70% of a column's missingness is explained by the
# shared block propensity, 30% is idiosyncratic.
# ─────────────────────────────────────────────────────────────────

BLOCK_ALPHA = 0.7   # within-block correlation strength

blocks = {
    'insurance':    MISS_BLOCK_INSURANCE,
    'suicide':      MISS_BLOCK_SUICIDE,
    'demographics': MISS_BLOCK_DEMOGRAPHICS,
}

for block_name, block_cols in blocks.items():
    # Shared latent propensity per row for this block
    u_block = rng_miss.random(n_rows)

    for col in block_cols:
        r = base_rates[col]
        # Shared component: row is missing if block propensity falls below rate
        shared_miss = u_block < r
        # Idiosyncratic component: independent draw, scaled down by (1 - alpha)
        idio_miss   = rng_miss.random(n_rows) < (r * (1 - BLOCK_ALPHA))
        mask        = shared_miss | idio_miss

        if enc_df[col].dtype == object:
            enc_df[col] = enc_df[col].astype(object)
            enc_df.loc[mask, col] = None
        else:
            enc_df.loc[mask, col] = np.nan

    print(f"  Block '{block_name}': {len(block_cols)} columns  "
          f"(alpha={BLOCK_ALPHA}, mean base rate "
          f"{np.mean([base_rates[c] for c in block_cols]):.3f}).")

# ─────────────────────────────────────────────────────────────────
# VALIDATION — report actual missing rates per column and mechanism
# ─────────────────────────────────────────────────────────────────

print("\n  Missingness summary (target vs actual):")
print(f"  {'Column':<30} {'Mechanism':<8} {'Target':>7} {'Actual':>7}")
print(f"  {'-'*56}")

mech_map = (
    [(c, 'MCAR')  for c in MISS_MCAR] +
    [(c, 'MAR')   for c in mar_cols] +
    [(c, 'Block') for block in blocks.values() for c in block]
)

for col, mech in mech_map:
    target = base_rates[col]
    if enc_df[col].dtype == object:
        actual = enc_df[col].isna().mean()
    else:
        actual = enc_df[col].isna().mean()
    print(f"  {col:<30} {mech:<8} {target:>6.3f}  {actual:>6.3f}")

# ═════════════════════════════════════════════════════════════════
# SECTION 10 — EDGE CASES
# Patients designed to stress-test downstream preprocessing:
# single encounters, high utilizers, boundary dates, age extremes,
# and a confirmed suicide attempt case.
# ═════════════════════════════════════════════════════════════════

print(f"Adding {N_EDGE_PATIENTS} edge case patients...")

next_id = enc_df['PatientDurableKey'].max() + 1

# Scale edge case types proportionally across N_EDGE_PATIENTS
# Fixed types: single encounter, boundary dates, all-null, young, old, suicide attempt = 6 minimum
# Scalable types: high utilizers, duplicate dates, ED-only, inpatient-only fill the rest
N_FIXED_EDGE    = 6
N_SCALABLE_EDGE = N_EDGE_PATIENTS - N_FIXED_EDGE
n_high_util  = max(1, N_SCALABLE_EDGE // 4)
n_dup_dates  = max(1, N_SCALABLE_EDGE // 4)
n_ed_only    = max(1, N_SCALABLE_EDGE // 4)
n_inpat_only = max(1, N_SCALABLE_EDGE - n_high_util - n_dup_dates - n_ed_only)

def base_enc(pid, enc_num, enc_date, enc_type,
             outpat_pri=None, outpat_nonpri=None, ed_pri=None):
    return {
        'PatientDurableKey': pid, 'EncounterKey': f"EC{pid}_E{enc_num}",
        'EncDate': enc_date, 'EncounterType': enc_type,
        'OutpatPrimaryDx': outpat_pri, 'OutpatNonPrimaryDx': outpat_nonpri,
        'EDPrimDx': ed_pri,
    }

edge_records = []

# 1. Single encounter patient
edge_records.append(base_enc(next_id, 0, pd.Timestamp('2021-06-15'),
                              'outpatient', outpat_pri='F33.3'))
next_id += 1

# 2. High utilizers
for _ in range(n_high_util):
    high_id = next_id; next_id += 1
    for i in range(50):
        edge_records.append(base_enc(high_id, i,
            pd.Timestamp('2018-01-01') + pd.Timedelta(days=i*30),
            'outpatient', outpat_pri='F33.3', outpat_nonpri='F41.1'))

# 3. Same-date duplicate encounters
for _ in range(n_dup_dates):
    dup_id = next_id; next_id += 1
    for i in range(3):
        edge_records.append(base_enc(dup_id, i, pd.Timestamp('2022-03-10'),
                                     'outpatient', outpat_pri='F32.3'))

# 4. Boundary date encounters
bnd_id = next_id; next_id += 1
edge_records.append(base_enc(bnd_id, 0, OBS_START, 'outpatient', outpat_pri='F41.1'))
edge_records.append(base_enc(bnd_id, 1, OBS_END,   'outpatient', outpat_pri='F41.1'))

# 5. All-null dx encounter
edge_records.append(base_enc(next_id, 0, pd.Timestamp('2020-08-20'), 'inpatient_other'))
next_id += 1

# 6. ED-only patients
for _ in range(n_ed_only):
    ed_id = next_id; next_id += 1
    for i in range(5):
        edge_records.append(base_enc(ed_id, i,
            pd.Timestamp('2019-01-01') + pd.Timedelta(days=i*45),
            'ed', ed_pri='R45.851'))

# 7. Inpatient-only patients
for _ in range(n_inpat_only):
    inp_id = next_id; next_id += 1
    for i in range(4):
        edge_records.append(base_enc(inp_id, i,
            pd.Timestamp('2020-05-01') + pd.Timedelta(days=i*60),
            'inpatient_other', outpat_nonpri='F33.3'))

# 8. Very young patient (age 5)
young_id = next_id; next_id += 1
edge_records.append(base_enc(young_id, 0, pd.Timestamp('2021-09-01'),
                              'outpatient', outpat_pri='F84.0'))

# 9. Very old patient (age 85)
old_id = next_id; next_id += 1
edge_records.append(base_enc(old_id, 0, pd.Timestamp('2021-09-01'),
                              'outpatient', outpat_pri='F33.1'))

# 10. Current suicide attempt patient
sa_id = next_id; next_id += 1
edge_records.append(base_enc(sa_id, 0, pd.Timestamp('2023-02-14'),
                              'ed', ed_pri='R45.851'))

edge_df = pd.DataFrame(edge_records)

# Fill all binary variables with 'N' defaults, then override specific cases
for col in binary_vars:
    edge_df[col] = 'N'
edge_df.loc[edge_df['PatientDurableKey'] == sa_id, 'CurrentSuicideAttempt'] = 'Y'

# Fill ordinal / continuous columns
for score_name, cfg in score_configs.items():
    probs = np.array(cfg['level_probs']); probs /= probs.sum()
    edge_df[f'{score_name}Level']   = np.random.choice(RISK_LEVELS, size=len(edge_df), p=probs)
    edge_df[f'{score_name}Score']   = generate_continuous(len(edge_df), cfg['score_mean'], cfg['score_sd'], cfg['score_low'], cfg['score_high'], cfg['score_missing'])
    edge_df[f'{score_name}DateKey'] = generate_skewed_years(len(edge_df), OBS_START, OBS_END, cfg['date_missing'])

for col in ['PainIntensityScore', 'PainIntensityDateKey', 'VASScore', 'VASDateKey',
            'FatigueScore', 'FatigueDateKey', 'PainInterfScore', 'PainInterfDateKey']:
    edge_df[col] = np.nan

for col, mr in lab_missing_rates.items():
    edge_df[col] = generate_skewed_years(len(edge_df), OBS_START, OBS_END, mr)

for col in ['PsychiatryVisitCount', 'PsychologyVisitCount', 'EDVisitCount', 'PxCount']:
    edge_df[col] = 0.0

# Demographics — defaults, then override age extremes
edge_df['AgeInYears']            = 35.0
edge_df['EncounterHeightInches'] = 66.0
edge_df['EncounterWeightLbs']    = 165.0
edge_df['EncounterBMI']          = 26.6
edge_df.loc[edge_df['PatientDurableKey'] == young_id,
            ['AgeInYears','EncounterHeightInches','EncounterWeightLbs','EncounterBMI']] = [5.0, 43.0, 40.0, 15.8]
edge_df.loc[edge_df['PatientDurableKey'] == old_id,
            ['AgeInYears','EncounterHeightInches','EncounterWeightLbs','EncounterBMI']] = [85.0, 63.0, 142.0, 25.2]

enc_df = pd.concat([enc_df, edge_df], ignore_index=True)
enc_df = enc_df.sort_values(['PatientDurableKey', 'EncDate']).reset_index(drop=True)
print(f"  Edge cases added. Total encounters: {len(enc_df):,}")

# ═════════════════════════════════════════════════════════════════
# SECTION 11 — COLUMN ORDERING & SAVE
# ═════════════════════════════════════════════════════════════════

print("Reordering columns and saving...")

enc_order = [
    'PatientDurableKey', 'EncounterKey',
    'EncDate', 'EncounterType',
    'AgeInYears', 'EncounterWeightLbs',
    'WeightGain10_20lbs', 'WeightGain20pluslbs',
    'WeightLoss10_20lbs', 'WeightLoss20pluslbs',
    'EncounterHeightInches', 'EncounterBMI',
    'HealthInsLoss', 'MedicareYN', 'MedicaidYN', 'SelfPayYN', 'MiscOtherYN',
    'PsychiatryVisitCount', 'PsychologyVisitCount',
    'OutpatPrimaryDx', 'OutpatNonPrimaryDx',
    'HxSuicideAttempt', 'HxSuicideAttempt30DaysPrior',
    'Opioids', 'SSRI', 'Antipsychotics', 'Lithium', 'Anticonvulsants',
    'Benzos', 'NRI', 'ADHDStimulants',
    'Score5Score',  'Score5Level',  'Score5DateKey',
    'Score49Score', 'Score49Level', 'Score49DateKey',
    'Score50Score', 'Score50Level', 'Score50DateKey',
    'Score51Score', 'Score51Level', 'Score51DateKey',
    'Score52Score', 'Score52Level', 'Score52DateKey',
    'Score53Score', 'Score53Level', 'Score53DateKey',
    'Score54Score', 'Score54Level', 'Score54DateKey',
    'Depression', 'DepressionPlus', 'MDDActive', 'MDDRemission',
    'AnxietyPlus', 'Anxiety', 'AdjStressDisorder', 'MoodDisorders',
    'PsychoticPlus', 'Schizophrenia', 'PersDisorders', 'MentalDisorder',
    'EatingDisorder', 'SexualDysfunction', 'SleepDisorder', 'Autism',
    'PTSD', 'PTSDPlus', 'ImpulseDisorders', 'ExternalizingDis', 'ADHD',
    'MalignantNeoplasm',
    'EDVisitCount', 'EDPrimDx',
    'PainIntensityScore', 'PainIntensityDateKey',
    'VASScore',           'VASDateKey',
    'FatigueScore',       'FatigueDateKey',
    'PainInterfScore',    'PainInterfDateKey',
    'PxCount',
    'PrescripMed45', 'PrescripMedGT5',
    'Pulmonary', 'FastingGlucose', 'SpotGlucose', 'LipidLevels',
    'CRP', 'Cortisol', 'Thyroid',
    'CurrentSuicideAttempt',
]

enc_df = enc_df[enc_order]
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
enc_df.to_parquet(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print(f"Shape: {enc_df.shape}")

# ═════════════════════════════════════════════════════════════════
# SECTION 12 — VALIDATION
# ═════════════════════════════════════════════════════════════════

print("\n── Validation ───────────────────────────────────────")
print(f"Total encounters:  {len(enc_df):,}")
print(f"Total patients:    {enc_df['PatientDurableKey'].nunique():,}")
print(f"Total columns:     {len(enc_df.columns)}")
print(f"Date range:        {enc_df['EncDate'].min().date()} to {enc_df['EncDate'].max().date()}")

enc_per_pat = enc_df.groupby('PatientDurableKey')['EncounterKey'].nunique()
print(f"\nEncounters/patient — Mean: {enc_per_pat.mean():.2f}  SD: {enc_per_pat.std():.2f}")
print(f"  (target Mean: {ENC_MEAN}  SD: {ENC_SD})")

print(f"\nEncounter type split (target in parentheses):")
type_map = {'outpat_only':'outpatient','ed_only':'ed','both':'both','neither':'inpatient_other'}
for etype, target in zip(ENC_TYPES, ENC_TYPE_P):
    actual = (enc_df['EncounterType'] == type_map[etype]).mean() * 100
    print(f"  {etype:<15} {actual:.1f}%  (target {target*100:.1f}%)")

print(f"\nCurrentSuicideAttempt (Y): {(enc_df['CurrentSuicideAttempt']=='Y').mean():.3f}  (target ~0.040)")
print(f"HxSuicideAttempt (Y):      {(enc_df['HxSuicideAttempt']=='Y').mean():.3f}  (target ~0.180)")

try:
    # ── Copula validation ─────────────────────────────────────────────
    print("\n── Copula Validation ────────────────────────────────────────")
    
    # Pick one variable from each block to keep output readable
    sample_vars = [
        'EncounterBMI',          # anthropometric
        'MedicaidYN',            # insurance
        'PsychiatryVisitCount',  # utilization
        'HxSuicideAttempt',      # suicide
        'SSRI',                  # medications
        'MDDActive',             # mood
        'Anxiety',               # anxiety
        'Schizophrenia',         # psychotic
        'ADHD',                  # neurodevelop
        'SleepDisorder',         # other_dx
        'Score52Score',          # scores
        'FastingGlucose',        # labs
    ]
    
    # Convert Y/N to numeric for correlation
    val_df = enc_df[sample_vars].copy()
    for col in val_df.columns:
        if val_df[col].dtype == object:
            val_df[col] = (val_df[col] == 'Y').astype(float)
    
    corr_matrix = val_df.select_dtypes(include='number').corr().round(2)
    print(corr_matrix)
    print(f"\nMean within-block correlation (expected ~{WITHIN_BLOCK_CORR}):")
    print(f"  MDDActive  <-> Depression : {val_df['MDDActive'].corr(enc_df['Depression'].map({'Y':1,'N':0})):.2f}")
    print(f"  Anxiety    <-> AnxietyPlus: {val_df['Anxiety'].corr(enc_df['AnxietyPlus'].map({'Y':1,'N':0})):.2f}")
    print(f"\nMean between-block correlation (expected ~{BETWEEN_BLOCK_CORR}):")
    print(f"  EncounterBMI <-> SSRI     : {val_df['EncounterBMI'].corr(enc_df['SSRI'].map({'Y':1,'N':0})):.2f}")
    print(f"  FastingGlucose <-> ADHD   : {val_df['FastingGlucose'].corr(enc_df['ADHD'].map({'Y':1,'N':0})):.2f}")
    
except Exception as e:
    print(f"Copula validation skipped: {e}")
