#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# ═════════════════════════════════════════════════════════════════
# SECTION 1 — PARAMETERS & CONSTANTS
# ═════════════════════════════════════════════════════════════════
ENC_PATH = "/uufs/chpc.utah.edu/common/PE/proj_synthetic_EHR/Monika_Workspace/00_Data/00_Raw_Data/Encounter_full.parquet"
OUT_PATH = "/uufs/chpc.utah.edu/common/PE/proj_synthetic_EHR/Monika_Workspace/00_Data/00_Raw_Data/Patient_full.parquet"

OBS_START = pd.Timestamp('2016-01-01')
OBS_END   = pd.Timestamp('2024-12-31')

np.random.seed(42)

# ── String label maps for all nominal variables ──
SEX_MAP              = {0: 'Female', 1: 'Male', 2: 'Other'}
SEX_AT_BIRTH_MAP     = {0: 'Female', 1: 'Male', 2: 'Uncertain', 3: 'Not recorded on birth certificate'}
GENDER_IDENTITY_MAP  = {0: 'Female', 1: 'Male',
                        2: 'Transgender Male / Female-to-Male',
                        3: 'Transgender Female / Male-to-Female'}
LANGUAGE_MAP         = {0: 'Arabic', 1: 'Chinese', 2: 'English',
                        3: 'Haitian; Haitian Creole', 4: 'Japanese',
                        5: 'Korean', 6: 'Other', 7: 'Portuguese',
                        8: 'Russian', 9: 'Spanish; Castilian', 10: 'Vietnamese'}
STATE_MAP            = {
    0:'Alabama', 1:'Alaska', 2:'Arizona', 3:'Arkansas', 4:'California',
    5:'Colorado', 6:'Connecticut', 7:'Delaware', 8:'District of Columbia',
    9:'Florida', 10:'Georgia', 11:'Hawaii', 12:'Idaho', 13:'Illinois',
    14:'Indiana', 15:'Iowa', 16:'Kansas', 17:'Kentucky', 18:'Louisiana',
    19:'Maine', 20:'Maryland', 21:'Massachusetts', 22:'Michigan', 23:'Minnesota',
    24:'Mississippi', 25:'Missouri', 26:'Montana', 27:'Nebraska', 28:'Nevada',
    29:'New Hampshire', 30:'New Jersey', 31:'New Mexico', 32:'New York',
    33:'North Carolina', 34:'North Dakota', 35:'Ohio', 36:'Oklahoma',
    37:'Oregon', 38:'Pennsylvania', 39:'Puerto Rico', 40:'Rhode Island',
    41:'South Carolina', 42:'South Dakota', 43:'Tennessee', 44:'Texas',
    45:'Utah', 46:'Vermont', 47:'Virgin Islands', 48:'Virginia', 49:'Washington',
    50:'West Virginia', 51:'Wisconsin', 52:'Wyoming'
}
STATE_ABBREV_MAP     = {
    0:'AL', 1:'AK', 2:'AZ', 3:'AR', 4:'CA', 5:'CO', 6:'CT', 7:'DE',
    8:'DC', 9:'FL', 10:'GA', 11:'HI', 12:'ID', 13:'IL', 14:'IN',
    15:'IA', 16:'KS', 17:'KY', 18:'LA', 19:'ME', 20:'MD', 21:'MA',
    22:'MI', 23:'MN', 24:'MS', 25:'MO', 26:'MT', 27:'NE', 28:'NV',
    29:'NH', 30:'NJ', 31:'NM', 32:'NY', 33:'NC', 34:'ND', 35:'OH',
    36:'OK', 37:'OR', 38:'PA', 39:'PR', 40:'RI', 41:'SC', 42:'SD',
    43:'TN', 44:'TX', 45:'UT', 46:'VT', 47:'VI', 48:'VA', 49:'WA',
    50:'WV', 51:'WI', 52:'WY'
}
MARITAL_MAP          = {
    0:'Common Law', 1:'Divorced', 2:'Domestic partner', 3:'Legally Separated',
    4:'Married', 5:'Never Married', 6:'Polygamous', 7:'Unmarried', 8:'Widowed'
}
BIRTH_CONTROL_MAP    = {
    0:'Abstinence', 1:'Cervical cap', 2:'Coitus Interruptus',
    3:'Combined Oral Contraceptive Pill', 4:'Condom', 5:'Condom Female',
    6:'Condom Male', 7:'Depo-Provera', 8:'Diaphragm', 9:'Emergency Contraception',
    10:'Essure', 11:'Female Sterilization', 12:'Fertility Awareness',
    13:'Hysterectomy', 14:'I.U.D. - Copper', 15:'I.U.D. - Hormonal',
    16:'I.U.D. - Unspecified', 17:'Implant', 18:'Injection', 19:'Inserts',
    20:'Male Sterilization', 21:'Nexplanon', 22:'NuvaRing', 23:'Other',
    24:'Patch', 25:'Pill - Unspecified', 26:'Post-menopausal',
    27:'Progestin-only Pill', 28:'Ring', 29:'Spermicide', 30:'Sponge',
    31:'Surgical', 32:'Tubal Ligation', 33:'Tubal Occlusion', 34:'Vasectomy'
}
ABUSED_SUBSTANCE_MAP = {
    0:'"Crack" cocaine', 1:'Amphetamines', 2:'Amyl nitrate',
    3:'Anabolic steroids', 4:'Barbiturates', 5:'Benzodiazepines',
    6:'Cocaine', 7:'Codeine', 8:'Fentanyl', 9:'Flunitrazepam',
    10:'GHB', 11:'Hashish', 12:'Heroin', 13:'Hydrocodone',
    14:'Hydromorphone', 15:'IV', 16:'Ketamine', 17:'LSD',
    18:'MDMA (ecstacy)', 19:'Marijuana', 20:'Mescaline',
    21:'Methamphetamines', 22:'Methaqualone', 23:'Methodone',
    24:'Methylphenidate', 25:'Morphine', 26:'Nitrous oxide',
    27:'Opium', 28:'Other', 29:'Oxycodone', 30:'PCP', 31:'Psilocybin'
}
RACE_MAP             = {
    0: 'American Indian or Alaska Native', 1: 'Asian',
    2: 'Black or African American',
    3: 'Native Hawaiian or Other Pacific Islander',
    4: 'Other Race', 5: 'White'
}


# In[3]:


# ═════════════════════════════════════════════════════════════════
# SECTION 2 — HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def trunc_normal(mean, sd, low, high, n):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n)

def binary_col(p, n, missing_rate=0.0):
    vals = np.random.binomial(1, p, n).astype(float)
    if missing_rate > 0:
        vals[np.random.random(n) < missing_rate] = np.nan
    return vals

def ordinal_col(probs, labels, n, missing_rate=0.0):
    probs = np.array(probs, dtype=float) / np.sum(probs)
    vals  = np.random.choice(labels, size=n, p=probs).astype(object)
    if missing_rate > 0:
        vals[np.random.random(n) < missing_rate] = np.nan
    return vals

def nominal_col(probs, labels, n, missing_rate=0.0):
    """Alias for ordinal_col — use for unordered categories."""
    return ordinal_col(probs, labels, n, missing_rate)

def continuous_col(mean, sd, low, high, n, missing_rate=0.0):
    vals = trunc_normal(mean, sd, low, high, n).astype(float)
    if missing_rate > 0:
        vals[np.random.random(n) < missing_rate] = np.nan
    return vals

def skewed_date(n, obs_start, obs_end, missing_rate):
    """Dates skewed exponentially toward obs_end; NaT where missing."""
    obs_days = (obs_end - obs_start).days
    offsets  = np.clip(np.random.exponential(obs_days * 0.3, n).astype(int), 0, obs_days)
    dates    = [obs_end - pd.Timedelta(days=int(o)) for o in offsets]
    if missing_rate > 0:
        mask  = np.random.random(n) < missing_rate
        dates = [pd.NaT if mask[i] else dates[i] for i in range(n)]
    return dates

def paired_datekey(content_array, n, obs_start, obs_end):
    """Generate a DateKey that is NaT wherever its content variable is null."""
    return [
        skewed_date(1, obs_start, obs_end, 0.0)[0]
        if not _is_null(content_array[i]) else pd.NaT
        for i in range(n)
    ]

def _is_null(x):
    """Unified null check for both float NaN and object None."""
    if x is None:
        return True
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return pd.isna(x)

def map_labels(arr, label_map):
    """Map integer codes to string labels, preserving nulls."""
    return np.array([
        label_map.get(int(x)) if not _is_null(x) else None
        for x in arr
    ], dtype=object)


# In[4]:


# ═════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD ENCOUNTER DATA & BUILD PATIENT SPINE
# ═════════════════════════════════════════════════════════════════

print("Loading encounter data and building patient spine...")
enc_df = pd.read_parquet(ENC_PATH)
enc_df['EncDate'] = pd.to_datetime(enc_df['EncDate'])

patient_spine = (
    enc_df.sort_values('EncDate')
    .groupby('PatientDurableKey')
    .agg(
        first_enc_date=('EncDate', 'first'),
        last_enc_date=('EncDate',  'last'),
        age_at_first_enc=('AgeInYears', 'first'),
    )
    .reset_index()
)

N = len(patient_spine)
print(f"  {N} patients")

# ── BirthDate: back-calculated from age at first encounter ──
def backfill_birthdate(row):
    if pd.isna(row['age_at_first_enc']) or pd.isna(row['first_enc_date']):
        return pd.NaT
    whole_years  = int(row['age_at_first_enc'])
    extra_days   = int(round((row['age_at_first_enc'] - whole_years) * 365.25))
    approx_birth = (row['first_enc_date']
                    - pd.DateOffset(years=whole_years)
                    - pd.Timedelta(days=extra_days))
    return approx_birth + pd.Timedelta(days=int(np.random.randint(-30, 31)))

patient_spine['BirthDate'] = patient_spine.apply(backfill_birthdate, axis=1)

# ── DeathDate: ~3% of patients die after their last encounter ──
has_death   = np.random.random(N) < 0.03
death_dates = []
for i in range(N):
    if has_death[i]:
        last         = patient_spine['last_enc_date'].iloc[i]
        days_remain  = max(1, (OBS_END - last).days)
        death_dates.append(last + pd.Timedelta(days=int(np.random.randint(1, days_remain + 1))))
    else:
        death_dates.append(pd.NaT)
patient_spine['DeathDate'] = death_dates

pat = patient_spine[['PatientDurableKey', 'BirthDate', 'DeathDate']].copy()


# In[5]:


# ═════════════════════════════════════════════════════════════════
# SECTION 4 — DEMOGRAPHICS
# All nominal variables generated with string labels directly.
# ═════════════════════════════════════════════════════════════════

print("Generating demographics...")

# ── Race: generate as combined label first, then split into 5 columns ──
race_probs  = [0.62, 0.14, 0.06, 0.02, 0.01, 0.08, 0.07]
race_labels = list(RACE_MAP.values()) + ['Multiple Races']
race_combined = nominal_col(race_probs, race_labels, N, missing_rate=0.04)

first_race = np.full(N, None, dtype=object)
second_race = np.full(N, None, dtype=object)
third_race  = np.full(N, None, dtype=object)
fourth_race = np.full(N, None, dtype=object)
fifth_race  = np.full(N, None, dtype=object)

for i, val in enumerate(race_combined):
    if pd.isna(val):
        pass
    elif val == 'Multiple Races':
        chosen = np.random.choice(list(RACE_MAP.values()), size=2, replace=False)
        first_race[i], second_race[i] = chosen[0], chosen[1]
    else:
        first_race[i] = val

pat['FirstRace']  = first_race
pat['SecondRace'] = second_race
pat['ThirdRace']  = third_race
pat['FourthRace'] = fourth_race
pat['FifthRace']  = fifth_race

# ── Other demographics ──
pat['Ethnicity'] = binary_col(0.12, N, missing_rate=0.04)

sex_codes = nominal_col([0.55, 0.43, 0.02], list(SEX_MAP.keys()), N, missing_rate=0.02)
pat['Sex'] = map_labels(sex_codes, SEX_MAP)

sab_codes = nominal_col([0.55, 0.43, 0.01, 0.01], list(SEX_AT_BIRTH_MAP.keys()), N, missing_rate=0.03)
pat['SexAssignedAtBirth'] = map_labels(sab_codes, SEX_AT_BIRTH_MAP)

gi_codes = nominal_col([0.54, 0.42, 0.02, 0.02], list(GENDER_IDENTITY_MAP.keys()), N, missing_rate=0.08)
pat['GenderIdentity'] = map_labels(gi_codes, GENDER_IDENTITY_MAP)

lang_probs = [0.005, 0.010, 0.870, 0.003, 0.003, 0.003, 0.040, 0.010, 0.010, 0.040, 0.006]
lang_codes = nominal_col(lang_probs, list(LANGUAGE_MAP.keys()), N, missing_rate=0.05)
pat['PreferredLanguage'] = map_labels(lang_codes, LANGUAGE_MAP)

state_pop = np.array([
    4.9, 0.7, 7.3, 3.0, 39.5, 5.8, 3.6, 1.0, 0.7, 22.6,
    10.9, 1.4, 1.9, 12.7, 6.8, 3.2, 2.9, 4.5, 4.6, 1.4,
    6.2, 7.0, 10.0, 5.7, 3.0, 6.2, 1.1, 2.0, 3.1, 1.4,
    9.3, 2.1, 19.7, 10.7, 0.8, 11.8, 4.0, 4.2, 13.1, 3.2,
    1.1, 5.3, 0.9, 7.1, 30.5, 3.3, 0.6, 0.1, 8.7, 7.8,
    1.8, 5.9, 0.6
])
state_codes = nominal_col(state_pop / state_pop.sum(), list(STATE_MAP.keys()), N, missing_rate=0.06)
pat['ValidatedStateOrProvince_X']            = map_labels(state_codes, STATE_MAP)
pat['ValidatedStateOrProvinceAbbreviation_X']= map_labels(state_codes, STATE_ABBREV_MAP)

marital_probs = [0.02, 0.14, 0.03, 0.03, 0.32, 0.34, 0.00, 0.08, 0.04]
marital_codes = nominal_col(marital_probs, list(MARITAL_MAP.keys()), N, missing_rate=0.10)
pat['MaritalStatus'] = map_labels(marital_codes, MARITAL_MAP)


# In[6]:


# ═════════════════════════════════════════════════════════════════
# SECTION 5 — SOCIAL VULNERABILITY INDEX (SVI) & ADI
# ═════════════════════════════════════════════════════════════════

print("Generating SVI / ADI...")

svi_base_2018 = trunc_normal(50, 25, 0, 100, N)
svi_base_2020 = np.clip(svi_base_2018 + np.random.normal(0, 5, N), 0, 100)

def svi_theme(base, noise_sd, missing_rate):
    vals = np.clip(base + np.random.normal(0, noise_sd, N), 0, 100).round(4)
    vals[np.random.random(N) < missing_rate] = np.nan
    return vals

pat['SviHouseholdCharacteristicsPctlRankByZip2020_X']    = svi_theme(svi_base_2020, 12, 0.22)
pat['SviHouseholdCompositionPctlRankingByZip2018_X']     = svi_theme(svi_base_2018, 12, 0.24)
pat['SviHousingTypeTransportationPctlRankByZip2020_X']   = svi_theme(svi_base_2020, 15, 0.23)
pat['SviHousingTypeTransportationPctlRankingByZip2018_X']= svi_theme(svi_base_2018, 15, 0.25)
pat['SviMinorityStatusLanguagePctlRankingByZip2018_X']   = svi_theme(svi_base_2018, 18, 0.24)
pat['SviOverallPctlRankByZip2020_X']                     = svi_theme(svi_base_2020,  8, 0.22)
pat['SviOverallPctlRankingByZip2018_X']                  = svi_theme(svi_base_2018,  8, 0.24)
pat['SviRacialEthnicMinorityStatusPctlRankByZip2020_X']  = svi_theme(svi_base_2020, 18, 0.22)
pat['SviSocioeconomicPctlRankByZip2020_X']               = svi_theme(svi_base_2020, 12, 0.22)
pat['SviSocioeconomicPctlRankingByZip2018_X']            = svi_theme(svi_base_2018, 12, 0.24)
pat['ADIUSPercentileRank'] = continuous_col(55, 28, 1, 100, N, missing_rate=0.28)


# In[7]:


# ═════════════════════════════════════════════════════════════════
# SECTION 6 — SDOH / LIFESTYLE FLOWSHEET VARIABLES
# Each content variable is immediately followed by its DateKey.
# Content variables use string labels where applicable.
# ═════════════════════════════════════════════════════════════════

print("Generating SDOH / lifestyle flowsheet variables...")

def add_pair(col_name, content_array):
    """Add content column and its paired DateKey to pat."""
    pat[col_name]            = content_array
    pat[f'{col_name}DateKey'] = paired_datekey(content_array, N, OBS_START, OBS_END)

# ── Food security ──
add_pair('FoodWorry',   ordinal_col([0.60, 0.28, 0.12], [0,1,2], N, 0.55))
add_pair('AlcDrinksPerDay', continuous_col(1.2, 1.5, 0, 10, N, missing_rate=0.55))
add_pair('HistoryAlcUse',   continuous_col(0.5, 1.0, 0, 5,  N, missing_rate=0.50))

# ── Communicable disease exposure (content + date) ──
_cde = binary_col(0.08, N, missing_rate=0.78)
add_pair('CommDiseaseExp', _cde)

# ── Social connections ──
add_pair('SocConnMember', binary_col(0.45, N, missing_rate=0.68))
add_pair('SocConnPhone',  ordinal_col([0.12,0.10,0.16,0.22,0.40], [0,1,2,3,4], N, 0.68))
add_pair('TransportMed',  binary_col(0.12, N, missing_rate=0.58))
add_pair('CigPackYears',  continuous_col(8, 10, 0, 60, N, missing_rate=0.55))

# ── Birth control ──
bc_probs = np.ones(35)
bc_probs[3] *= 5; bc_probs[15] *= 4; bc_probs[17] *= 3
bc_probs[26] *= 2; bc_probs[32] *= 3; bc_probs[4] *= 2
bc_probs /= bc_probs.sum()
bc_codes = ordinal_col(bc_probs, list(BIRTH_CONTROL_MAP.keys()), N, missing_rate=0.62)
add_pair('BirthControl', map_labels(bc_codes, BIRTH_CONTROL_MAP))

# ── Food scarcity ──
add_pair('FoodScarcity', ordinal_col([0.62, 0.26, 0.12], [0,1,2], N, 0.55))
add_pair('SocConnGetTog', ordinal_col([0.08,0.15,0.22,0.28,0.27], [0,1,2,3,4], N, 0.68))

# ── IPV ──
add_pair('IPVPhysAbuse',   binary_col(0.08, N, missing_rate=0.65))
add_pair('TransportNonMed', binary_col(0.10, N, missing_rate=0.58))

# ── FreqDrugMisuse ──
add_pair('FreqDrugMisuse', ordinal_col([0.60,0.20,0.12,0.08], [0,1,2,3], N, 0.65))

# ── Smoking ──
add_pair('CigPacksPerDay', continuous_col(0.4, 0.5, 0, 3, N, missing_rate=0.55))
add_pair('SmokingStatus',  ordinal_col([0.52,0.18,0.06,0.08,0.12,0.04], [0,1,2,3,4,5], N, 0.35))
add_pair('PhysActivityDPW', ordinal_col([0.18,0.10,0.12,0.15,0.12,0.12,0.10,0.11], [0,1,2,3,4,5,6,7], N, 0.58))
add_pair('Stress',         ordinal_col([0.10,0.18,0.28,0.26,0.18], [0,1,2,3,4], N, 0.55))

# ── IPV continued ──
add_pair('IPVEmotional',   binary_col(0.12, N, missing_rate=0.65))
add_pair('SocConnChurch',  ordinal_col([0.30,0.28,0.42], [0,1,2], N, 0.68))

# ── Physical activity minutes ──
pamps_probs = np.array([0.18] + [0.055]*15); pamps_probs /= pamps_probs.sum()
add_pair('PhysActivityMPS', ordinal_col(pamps_probs, list(range(16)), N, 0.58))

# ── Smokeless / substance ──
add_pair('SmokelessStatus', ordinal_col([0.72,0.16,0.12], [0,1,2], N, 0.65))

abused_probs = np.array([0.5,2,0.2,0.3,0.8,2.5,1.5,0.8,1.2,0.3,
                          0.2,0.5,1.0,0.5,0.3,0.2,0.3,0.2,1.5,8.0,
                          0.2,2.0,0.2,0.5,0.2,0.5,0.2,0.2,1.5,1.0,
                          0.2,0.3])
abused_probs /= abused_probs.sum()
ab_codes = ordinal_col(abused_probs, list(ABUSED_SUBSTANCE_MAP.keys()), N, missing_rate=0.60)
add_pair('AbusedSubstance', map_labels(ab_codes, ABUSED_SUBSTANCE_MAP))

# ── SocConnLiving (content + date) ──
add_pair('SocConnLiving', ordinal_col([0.25,0.75], [0,1], N, 0.68))

# ── Sexual activity ──
add_pair('SexuallyActive', ordinal_col([0.08,0.25,0.67], [0,1,2], N, 0.60))
add_pair('AlcStdDrinks',   ordinal_col([0.30,0.30,0.20,0.12,0.05,0.03], [0,1,2,3,4,5], N, 0.55))
add_pair('IPVFear',        binary_col(0.10, N, missing_rate=0.65))
add_pair('Financial',      ordinal_col([0.30,0.25,0.22,0.15,0.08], [0,1,2,3,4], N, 0.50))

# ── TobaccoUse ──
add_pair('TobaccoUse', ordinal_col([0.55,0.25,0.20], [0,1,2], N, 0.40))

# ── Travel history ──
add_pair('TravelHistory', binary_col(0.15, N, missing_rate=0.75))

# ── Alcohol ──
add_pair('AlcoholFreq',  ordinal_col([0.35,0.22,0.18,0.15,0.10], [0,1,2,3,4], N, 0.50))
add_pair('AlcoholBinge', ordinal_col([0.45,0.22,0.15,0.12,0.06], [0,1,2,3,4], N, 0.50))
add_pair('IPVSexualAbuse', binary_col(0.04, N, missing_rate=0.65))
add_pair('SubstUseStatus', ordinal_col([0.35,0.28,0.15,0.22], [0,1,2,3], N, 0.55))
add_pair('SocConnMeetings', ordinal_col([0.35,0.28,0.37], [0,1,2], N, 0.68))
add_pair('SexualPartner', binary_col(0.55, N, missing_rate=0.70))
add_pair('HousingPlaceLived', continuous_col(1.3, 0.8, 1, 10, N, missing_rate=0.60))
add_pair('HousingHomeless',   binary_col(0.07, N, missing_rate=0.62))
add_pair('HousingMortgage',   binary_col(0.10, N, missing_rate=0.62))


# In[8]:


# ═════════════════════════════════════════════════════════════════
# SECTION 7 — LIFETIME CLINICAL CONDITIONS
# Binary Y/N flags; prevalences from behavioral health literature.
# ═════════════════════════════════════════════════════════════════

print("Generating lifetime clinical conditions...")

clinical_conditions = {
    'AcutePain': 0.18, 'Arthropathies': 0.15, 'Cardiovascular': 0.22,
    'Chlamydia': 0.04, 'ChronicFatigue': 0.08, 'ChronicPain': 0.25,
    'Covid19': 0.18, 'Dementing': 0.04, 'Diabetes': 0.14, 'GaitImp': 0.05,
    'Gonorrhea': 0.02, 'Herpes': 0.05, 'HIV': 0.03, 'Neoplasms': 0.08,
    'Malnutrition': 0.03, 'NervousSys': 0.14, 'Obesity': 0.28,
    'PregnancyAbortive': 0.06, 'SleepApnea': 0.12, 'Syphilis': 0.02,
    'TBI': 0.10, 'Weakness': 0.06, 'Hospice': 0.02,
}

for col, prev in clinical_conditions.items():
    pat[col] = np.where(np.random.random(N) < prev, 'Y', 'N')

# ═════════════════════════════════════════════════════════════════
# SECTION 7.5 — GAUSSIAN COPULA (SYNTHETIC CORRELATIONS)
# Imposes a block correlation structure across all clinical variables.
# Correlations are synthetic and for pipeline testing only.
# Must run after Section 7 (all variables generated) and before
# Section 8 (SuicideAttempt derived from encounters — not touched here).
# ═════════════════════════════════════════════════════════════════

from scipy.stats import norm

WITHIN_BLOCK_CORR  = 0.8
BETWEEN_BLOCK_CORR = 0.15

# ── Define variable blocks ────────────────────────────────────────
copula_blocks = {
    'demographics':    ['Ethnicity', 'Sex', 'SexAssignedAtBirth',
                        'GenderIdentity', 'PreferredLanguage',
                        'ValidatedStateOrProvince_X', 'MaritalStatus'],
    'race':            ['FirstRace', 'SecondRace'],
    'svi_2020':        ['SviOverallPctlRankByZip2020_X',
                        'SviSocioeconomicPctlRankByZip2020_X',
                        'SviHouseholdCharacteristicsPctlRankByZip2020_X',
                        'SviHousingTypeTransportationPctlRankByZip2020_X',
                        'SviRacialEthnicMinorityStatusPctlRankByZip2020_X'],
    'svi_2018':        ['SviOverallPctlRankingByZip2018_X',
                        'SviSocioeconomicPctlRankingByZip2018_X',
                        'SviHousingTypeTransportationPctlRankingByZip2018_X',
                        'SviHouseholdCompositionPctlRankingByZip2018_X',
                        'SviMinorityStatusLanguagePctlRankingByZip2018_X',
                        'ADIUSPercentileRank'],
    'food_finance':    ['FoodWorry', 'FoodScarcity', 'Financial'],
    'housing':         ['HousingPlaceLived', 'HousingHomeless', 'HousingMortgage'],
    'transport':       ['TransportMed', 'TransportNonMed'],
    'ipv':             ['IPVPhysAbuse', 'IPVEmotional', 'IPVFear', 'IPVSexualAbuse'],
    'social':          ['SocConnMember', 'SocConnPhone', 'SocConnGetTog',
                        'SocConnChurch', 'SocConnMeetings', 'SocConnLiving'],
    'smoking':         ['SmokingStatus', 'SmokelessStatus', 'TobaccoUse',
                        'CigPacksPerDay', 'CigPackYears'],
    'alcohol':         ['HistoryAlcUse', 'AlcDrinksPerDay', 'AlcoholFreq',
                        'AlcoholBinge', 'AlcStdDrinks'],
    'substance':       ['SubstUseStatus', 'AbusedSubstance',
                        'FreqDrugMisuse', 'CommDiseaseExp'],
    'activity_stress': ['PhysActivityDPW', 'PhysActivityMPS', 'Stress'],
    'sexual':          ['SexuallyActive', 'SexualPartner',
                        'BirthControl', 'TravelHistory'],
    'clinical':        ['AcutePain', 'Arthropathies', 'Cardiovascular',
                        'Chlamydia', 'ChronicFatigue', 'ChronicPain',
                        'Covid19', 'Dementing', 'Diabetes', 'GaitImp',
                        'Gonorrhea', 'Herpes', 'HIV', 'Neoplasms',
                        'Malnutrition', 'NervousSys', 'Obesity',
                        'PregnancyAbortive', 'SleepApnea', 'Syphilis',
                        'TBI', 'Weakness', 'Hospice'],
}

# ── Classify each variable for back-transform ─────────────────────
yn_binary = list(clinical_conditions.keys())

numeric_binary = [
    'Ethnicity', 'HousingHomeless', 'HousingMortgage',
    'TransportMed', 'TransportNonMed', 'IPVPhysAbuse',
    'IPVEmotional', 'IPVFear', 'IPVSexualAbuse',
    'SocConnMember', 'SexualPartner', 'CommDiseaseExp', 'TravelHistory',
]

string_nominal = [
    'Sex', 'SexAssignedAtBirth', 'GenderIdentity', 'PreferredLanguage',
    'ValidatedStateOrProvince_X', 'MaritalStatus',
    'FirstRace', 'SecondRace', 'BirthControl', 'AbusedSubstance',
]

copula_vars = [v for block in copula_blocks.values() for v in block]
n_vars      = len(copula_vars)

# remaining cols are numeric ordinal/continuous — quantile back-transform
quantile_cols = [
    v for v in copula_vars
    if v not in yn_binary
    and v not in numeric_binary
    and v not in string_nominal
]

# ── Build correlation matrix ──────────────────────────────────────
R = np.full((n_vars, n_vars), BETWEEN_BLOCK_CORR)
np.fill_diagonal(R, 1.0)

idx_map = {v: i for i, v in enumerate(copula_vars)}
for block_vars in copula_blocks.values():
    for v1 in block_vars:
        for v2 in block_vars:
            if v1 != v2:
                R[idx_map[v1], idx_map[v2]] = WITHIN_BLOCK_CORR

# ── Ensure positive definiteness ──────────────────────────────────
eigvals, eigvecs = np.linalg.eigh(R)
eigvals = np.clip(eigvals, 1e-6, None)
R       = eigvecs @ np.diag(eigvals) @ eigvecs.T
d       = np.sqrt(np.diag(R))
R       = R / np.outer(d, d)

# ── Helpers ───────────────────────────────────────────────────────
def to_uniform(x):
    """
    Rank-based empirical CDF.
    - String columns are label-encoded before ranking.
    - NaN/None positions are filled with random uniform draws so the
      copula sees no gaps; original nulls are restored during back-transform.
    """
    x       = np.array(x, dtype=object)
    is_null = np.array([_is_null(v) for v in x])
    valid_x = x[~is_null]

    try:
        valid_numeric = valid_x.astype(float)
    except (ValueError, TypeError):
        cats          = {v: i for i, v in enumerate(sorted(set(valid_x.tolist())))}
        valid_numeric = np.array([cats[v] for v in valid_x], dtype=float)

    n             = len(valid_numeric)
    ranks         = pd.Series(valid_numeric).rank(method='average').values
    out           = np.full(len(x), np.nan)
    out[~is_null] = ranks / (n + 1)
    out[is_null]  = np.random.uniform(0, 1, is_null.sum())
    return out.astype(float)

def from_uniform_quantile(u_new, x_orig):
    """Quantile back-transform for numeric ordinal/continuous columns."""
    x_numeric   = pd.to_numeric(pd.Series(x_orig), errors='coerce')
    valid       = x_numeric.dropna().values
    if len(valid) == 0:
        return u_new
    sorted_vals = np.sort(valid)
    quantiles   = np.linspace(0, 1, len(sorted_vals))
    return np.interp(u_new, quantiles, sorted_vals)

def from_uniform_nominal(u_new, x_orig):
    """
    Map new uniform values back to string category labels,
    preserving the original marginal category frequencies.
    """
    x_orig  = np.array(x_orig, dtype=object)
    is_null = np.array([_is_null(v) for v in x_orig])
    vals    = x_orig[~is_null]
    if len(vals) == 0:
        return x_orig

    cats, counts = np.unique(vals, return_counts=True)
    probs        = counts / counts.sum()
    thresholds   = np.concatenate([[0], np.cumsum(probs)])

    result = np.full(len(u_new), None, dtype=object)
    for j, cat in enumerate(cats):
        mask         = (u_new >= thresholds[j]) & (u_new < thresholds[j + 1])
        result[mask] = cat
    result[result == None] = cats[-1]   # edge case: u_new == 1.0
    return result

def restore_nulls(new_vals, x_orig, is_yn=False, is_numeric_bin=False):
    """Re-apply the original missingness pattern after back-transform."""
    x_orig  = np.array(x_orig, dtype=object)
    is_null = np.array([_is_null(v) for v in x_orig])
    if is_yn:
        new_vals          = np.array(new_vals, dtype=object)
        new_vals[is_null] = None
    elif is_numeric_bin:
        new_vals          = np.array(new_vals, dtype=float)
        new_vals[is_null] = np.nan
    else:
        new_vals          = np.array(new_vals, dtype=object)
        new_vals[is_null] = None
    return new_vals

# ── Forward transform: variables → uniform ───────────────────────
U = np.column_stack([to_uniform(pat[v].values) for v in copula_vars])

# ── Uniform → standard normal ────────────────────────────────────
Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))

# ── Sample from Gaussian copula ───────────────────────────────────
L     = np.linalg.cholesky(R)
Z_new = (L @ np.random.standard_normal((n_vars, N))).T
U_new = norm.cdf(Z_new)

# ── Back-transform: uniform → original variable space ────────────
for i, var in enumerate(copula_vars):
    orig = pat[var].values

    if var in yn_binary:
        # Prevalence computed over non-null rows only
        n_valid    = (pat[var].isin(['Y', 'N'])).sum()
        prevalence = (pat[var] == 'Y').sum() / n_valid if n_valid > 0 else 0.0
        new_vals   = np.where(U_new[:, i] > (1 - prevalence), 'Y', 'N')
        pat[var]   = restore_nulls(new_vals, orig, is_yn=True)

    elif var in numeric_binary:
        # Prevalence computed over non-null rows only
        orig_series = pd.to_numeric(pd.Series(orig), errors='coerce')
        n_valid     = orig_series.notna().sum()
        prevalence  = (orig_series > 0.5).sum() / n_valid if n_valid > 0 else 0.0
        new_vals    = np.where(U_new[:, i] > (1 - prevalence), 1.0, 0.0)
        pat[var]    = restore_nulls(new_vals, orig, is_numeric_bin=True)

    elif var in string_nominal:
        new_vals = from_uniform_nominal(U_new[:, i], orig)
        pat[var] = restore_nulls(new_vals, orig)

    else:  # quantile cols — numeric ordinal or continuous
        new_vals     = from_uniform_quantile(U_new[:, i], orig)
        orig_numeric = pd.to_numeric(pd.Series(orig), errors='coerce').dropna()
        if len(orig_numeric) > 0 and (orig_numeric == orig_numeric.round()).all():
            new_vals = np.round(new_vals)
        pat[var] = restore_nulls(new_vals, orig)

print(f"Copula applied across {n_vars} variables in {len(copula_blocks)} blocks.")

# ── Cast numeric-valued object columns back to float ─────────────
for var in quantile_cols:
    pat[var] = pd.to_numeric(pat[var], errors='coerce')

# ═════════════════════════════════════════════════════════════════
# SECTION 7.6 — COPULA VALIDATION
# ═════════════════════════════════════════════════════════════════

print("\n── Copula Validation (Patient) ──────────────────────────────")

sample_vars = [
    'Ethnicity',                          # demographics
    'SviOverallPctlRankByZip2020_X',      # svi_2020
    'ADIUSPercentileRank',                # svi_2018
    'FoodWorry',                          # food_finance
    'HousingHomeless',                    # housing
    'IPVPhysAbuse',                       # ipv
    'SocConnMember',                      # social
    'SmokingStatus',                      # smoking
    'AlcoholFreq',                        # alcohol
    'SubstUseStatus',                     # substance
    'Stress',                             # activity_stress
    'SexuallyActive',                     # sexual
    'Diabetes',                           # clinical
    'Obesity',                            # clinical
]

# ── dtype diagnostic ──────────────────────────────────────────────
print("\nDtype diagnostic:")
for col in sample_vars:
    sample = pat[col].dropna().iloc[:3].tolist() if pat[col].notna().any() else []
    print(f"  {col:<40} dtype: {pat[col].dtype}  sample: {sample}")

# ── Correlation matrix ────────────────────────────────────────────
print("\nCorrelation matrix:")
val_df = pd.DataFrame(index=range(len(pat)))
for col in sample_vars:
    s = pat[col]
    if s.dtype == object:
        # Y/N binary → 0/1; other strings → category codes
        if set(s.dropna().unique()).issubset({'Y', 'N'}):
            val_df[col] = (s == 'Y').astype(float)
        else:
            codes = pd.Categorical(s).codes.astype(float)
            val_df[col] = pd.Series(codes).where(pd.Series(codes) >= 0, np.nan)
    else:
        val_df[col] = pd.to_numeric(s, errors='coerce')

print(val_df.corr().round(2).to_string())

# ── Within-block spot checks ──────────────────────────────────────
print(f"\nWithin-block spot checks (expected ~{WITHIN_BLOCK_CORR}):")

def to_numeric_safe(series):
    s = pat[series]
    if s.dtype == object:
        if set(s.dropna().unique()).issubset({'Y', 'N'}):
            return (s == 'Y').astype(float)
        return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(s, errors='coerce')

pairs = [
    ('FoodWorry',    'FoodScarcity'),
    ('IPVPhysAbuse', 'IPVEmotional'),
    ('SmokingStatus','CigPacksPerDay'),
    ('AlcoholFreq',  'AlcoholBinge'),
    ('Diabetes',     'Obesity'),
]
for v1, v2 in pairs:
    x1 = to_numeric_safe(v1)
    x2 = to_numeric_safe(v2)
    print(f"  {v1:<22} <-> {v2:<22} : {x1.corr(x2):.2f}")

# ── Between-block spot checks ─────────────────────────────────────
print(f"\nBetween-block spot checks (expected ~{BETWEEN_BLOCK_CORR}):")
cross_pairs = [
    ('Diabetes',     'IPVPhysAbuse'),
    ('SmokingStatus','SviOverallPctlRankByZip2020_X'),
    ('Obesity',      'SocConnMember'),
    ('AlcoholFreq',  'HousingHomeless'),
]
for v1, v2 in cross_pairs:
    x1 = to_numeric_safe(v1)
    x2 = to_numeric_safe(v2)
    print(f"  {v1:<22} <-> {v2:<22} : {x1.corr(x2):.2f}")

# ── Prevalence / distribution checks ─────────────────────────────
print(f"\nPrevalence / distribution checks:")
prev_checks = {
    'Diabetes':            ('yn',      0.14),
    'Obesity':             ('yn',      0.28),
    'HousingHomeless':     ('numeric', 0.07),
    'IPVPhysAbuse':        ('numeric', 0.08),
    'SmokingStatus':       ('mean',    None),
    'ADIUSPercentileRank': ('mean',    None),
}
for col, (kind, target) in prev_checks.items():
    if kind == 'yn':
        n_valid = pat[col].isin(['Y', 'N']).sum()
        actual  = (pat[col] == 'Y').sum() / n_valid if n_valid > 0 else np.nan
        print(f"  {col:<35} prevalence: {actual:.3f}  target: {target:.3f}")
    elif kind == 'numeric':
        s       = pd.to_numeric(pat[col], errors='coerce')
        n_valid = s.notna().sum()
        actual  = (s > 0.5).sum() / n_valid if n_valid > 0 else np.nan
        print(f"  {col:<35} prevalence: {actual:.3f}  target: {target:.3f}")
    else:
        actual = pd.to_numeric(pat[col], errors='coerce').mean()
        print(f"  {col:<35} mean: {actual:.2f}")

# ── String nominal category preservation ─────────────────────────
print(f"\nString nominal category preservation:")
for col in ['Sex', 'MaritalStatus', 'PreferredLanguage']:
    n_cats = pat[col].dropna().nunique()
    print(f"  {col:<35} unique categories: {n_cats}")


# In[9]:


# ═════════════════════════════════════════════════════════════════
# SECTION 8 — DERIVE SuicideAttempt FROM ENCOUNTER DATA
# Patient-level flag: Y if any encounter has CurrentSuicideAttempt = Y
# ═════════════════════════════════════════════════════════════════

print("Deriving SuicideAttempt from encounter data...")

suicide_ever = (
    enc_df.groupby('PatientDurableKey')['CurrentSuicideAttempt']
    .apply(lambda x: 'Y' if (x == 'Y').any() else 'N')
    .reset_index()
    .rename(columns={'CurrentSuicideAttempt': 'SuicideAttempt'})
)
pat = pat.merge(suicide_ever, on='PatientDurableKey', how='left')


# In[10]:


# ═════════════════════════════════════════════════════════════════
# SECTION 9 — EDGE CASES
# Ensure all nominal categories have at least one patient so the
# downstream encoding assert (set == set(range(n))) always passes.
# ═════════════════════════════════════════════════════════════════

print("Adding edge case patients for missing nominal categories...")

next_id = pat['PatientDurableKey'].max() + 1
edge_cases = []

nominal_checks = {
    'ValidatedStateOrProvince_X': list(STATE_MAP.values()),
    'MaritalStatus':              list(MARITAL_MAP.values()),
    'PreferredLanguage':          list(LANGUAGE_MAP.values()),
    'BirthControl':               list(BIRTH_CONTROL_MAP.values()),
    'AbusedSubstance':            list(ABUSED_SUBSTANCE_MAP.values()),
}

for col, expected in nominal_checks.items():
    actual  = set(pat[col].dropna().unique())
    missing = set(expected) - actual
    for val in sorted(missing):
        base_row = pat.iloc[0].copy()
        base_row['PatientDurableKey'] = next_id
        base_row[col] = val
        base_row['SuicideAttempt'] = 'N'
        edge_cases.append(base_row)
        next_id += 1

if edge_cases:
    pat = pd.concat([pat, pd.DataFrame(edge_cases)], ignore_index=True)
    print(f"  Added {len(edge_cases)} edge case patients. Total: {len(pat)}")
else:
    print("  No missing categories — no edge cases needed.")


# In[11]:


# ═════════════════════════════════════════════════════════════════
# SECTION 10 — COLUMN ORDERING & SAVE
# ═════════════════════════════════════════════════════════════════

print("Reordering columns and saving...")

pat_order = [
    'PatientDurableKey', 'BirthDate', 'DeathDate',
    'FirstRace', 'SecondRace', 'ThirdRace', 'FourthRace', 'FifthRace',
    'Ethnicity', 'Sex', 'SexAssignedAtBirth', 'GenderIdentity',
    'PreferredLanguage', 'ValidatedStateOrProvince_X',
    'ValidatedStateOrProvinceAbbreviation_X', 'MaritalStatus',
    'SviHouseholdCharacteristicsPctlRankByZip2020_X',
    'SviHouseholdCompositionPctlRankingByZip2018_X',
    'SviHousingTypeTransportationPctlRankByZip2020_X',
    'SviHousingTypeTransportationPctlRankingByZip2018_X',
    'SviMinorityStatusLanguagePctlRankingByZip2018_X',
    'SviOverallPctlRankByZip2020_X',
    'SviOverallPctlRankingByZip2018_X',
    'SviRacialEthnicMinorityStatusPctlRankByZip2020_X',
    'SviSocioeconomicPctlRankByZip2020_X',
    'SviSocioeconomicPctlRankingByZip2018_X',
    'ADIUSPercentileRank',
    'FoodWorry',           'FoodWorryDateKey',
    'AlcDrinksPerDay',     'AlcDrinksPerDayDateKey',
    'HistoryAlcUse',       'HistoryAlcUseDateKey',
    'CommDiseaseExp',      'CommDiseaseExpDateKey',
    'SocConnMember',       'SocConnMemberDateKey',
    'SocConnPhone',        'SocConnPhoneDateKey',
    'TransportMed',        'TransportMedDateKey',
    'CigPackYears',        'CigPackYearsDateKey',
    'BirthControl',        'BirthControlDateKey',
    'FoodScarcity',        'FoodScarcityDateKey',
    'SocConnGetTog',       'SocConnGetTogDateKey',
    'IPVPhysAbuse',        'IPVPhysAbuseDateKey',
    'TransportNonMed',     'TransportNonMedDateKey',
    'FreqDrugMisuse',      'FreqDrugMisuseDateKey',
    'CigPacksPerDay',      'CigPacksPerDayDateKey',
    'SmokingStatus',       'SmokingStatusDateKey',
    'PhysActivityDPW',     'PhysActivityDPWDateKey',
    'Stress',              'StressDateKey',
    'IPVEmotional',        'IPVEmotionalDateKey',
    'SocConnChurch',       'SocConnChurchDateKey',
    'PhysActivityMPS',     'PhysActivityMPSDateKey',
    'SmokelessStatus',     'SmokelessStatusDateKey',
    'AbusedSubstance',     'AbusedSubstanceDateKey',
    'SocConnLiving',       'SocConnLivingDateKey',
    'SexuallyActive',      'SexuallyActiveDateKey',
    'AlcStdDrinks',        'AlcStdDrinksDateKey',
    'IPVFear',             'IPVFearDateKey',
    'Financial',           'FinancialDateKey',
    'TobaccoUse',          'TobaccoUseDateKey',
    'TravelHistory',       'TravelHistoryDateKey',
    'AlcoholFreq',         'AlcoholFreqDateKey',
    'AlcoholBinge',        'AlcoholBingeDateKey',
    'IPVSexualAbuse',      'IPVSexualAbuseDateKey',
    'SubstUseStatus',      'SubstUseStatusDateKey',
    'SocConnMeetings',     'SocConnMeetingsDateKey',
    'SexualPartner',       'SexualPartnerDateKey',
    'HousingPlaceLived',   'HousingPlaceLivedDateKey',
    'HousingHomeless',     'HousingHomelessDateKey',
    'HousingMortgage',     'HousingMortgageDateKey',
    'AcutePain', 'Arthropathies', 'Cardiovascular', 'Chlamydia',
    'ChronicFatigue', 'ChronicPain', 'Covid19', 'Dementing', 'Diabetes',
    'GaitImp', 'Gonorrhea', 'Herpes', 'HIV', 'Neoplasms', 'Malnutrition',
    'NervousSys', 'Obesity', 'PregnancyAbortive', 'SleepApnea', 'Syphilis',
    'TBI', 'Weakness', 'Hospice', 'SuicideAttempt',
]

pat = pat[pat_order]
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
pat.to_parquet(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print(f"Shape: {pat.shape}")


# In[12]:


# ═════════════════════════════════════════════════════════════════
# SECTION 11 — VALIDATION
# ═════════════════════════════════════════════════════════════════

print("\n── Validation ───────────────────────────────────────")
print(f"Patients:  {len(pat)}")
print(f"Columns:   {len(pat.columns)}")
print(f"BirthDate: {pat['BirthDate'].min().date()} to {pat['BirthDate'].max().date()}")

print("\nSex distribution:")
for v in ['Female', 'Male', 'Other']:
    pct = (pat['Sex'] == v).mean() * 100
    print(f"  {v:<8}  {pct:.1f}%")

print("\nSuicideAttempt (Y):", (pat['SuicideAttempt'] == 'Y').mean().round(3))

print("\nNominal coverage check:")
for col, expected in nominal_checks.items():
    actual  = set(pat[col].dropna().unique())
    missing = set(expected) - actual
    status  = '✓' if not missing else f'⚠ missing: {missing}'
    print(f"  {col:<45} {status}")

print("\nDateKey alignment (content null → date null):")
for col in ['FoodWorry', 'SmokingStatus', 'IPVPhysAbuse', 'Stress', 'TobaccoUse']:
    dk      = f'{col}DateKey'
    aligned = (pat[col].isna() == pat[dk].isna()).mean() * 100
    print(f"  {col:<25} {aligned:.1f}%  {'✓' if aligned > 95 else '⚠'}")

