import pandas as pd
import numpy as np
import re
import tldextract
import warnings
import unicodedata
import statsmodels.api as sm
from sklearn.model_selection import KFold
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import recordlinkage
from recordlinkage.preprocessing import clean
import math
import plotly.express as px

warnings.filterwarnings("ignore")

# ==========================================
# SECTION 1: CLEANING FUNCTIONS
# ==========================================

def load_and_initialize(file_path='startup_funding.csv'):
    try:
        df_original = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df_original)} records.")
        return df_original, df_original.copy()
    except FileNotFoundError:
        print("Error: startup_funding.csv not found. Please ensure it's in the directory.")
        return None, None

df_original, df = load_and_initialize()

# Initial Structural Cleaning
df.drop(['Sr No', 'Remarks', 'SubVertical'], axis=1, inplace=True, errors='ignore')

df = df.rename(columns={
    'InvestmentnType': 'Investment Type',
    'City  Location': 'City Location',
    'Date dd/mm/yyyy': 'Date'
})

def extract_startup_name(url):
    """
    Cleans Startup Names by extracting the domain using tldextract.
    This handles cases where URLs were listed instead of names.
    """
    if pd.isna(url) or url == "":
        return url

    ext = tldextract.extract(str(url))

    # If a domain is found, use it; otherwise, return the original string
    return ext.domain if ext.domain else url

# Applying to dataframe
df['Startup Name'] = df['Startup Name'].apply(extract_startup_name)

# Cleaning 'Amount in USD'
# Using regex to remove any character that isn't a digit, decimal, or negative sign
df['Amount in USD'] = (
    df['Amount in USD']
    .astype(str)
    .replace('nan', np.nan)
    .str.replace(r'[^0-9.\-]', '', regex=True)
)
df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')

# Standardizing Dates
# Converting to datetime objects to allow for time-series analysis later
date_col = 'Date'
df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')

# Quick validation of date parsing
unparsed_dates = df[date_col].isna().sum()
print(f"Finance Data Cleaning Complete.")
print(f"Date Parsing: {len(df) - unparsed_dates} successes, {unparsed_dates} failures.")

def clean_city_data(df, column_name='City Location'):
    # Basic string conversion and null handling
    df[column_name] = df[column_name].fillna('Not Specified').astype(str)

    # Fixing encoding artifacts (Byte Order Marks and Non-breaking spaces)
    df[column_name] = df[column_name].str.replace(r'\\xc2\\xa0', '', regex=True)
    df[column_name] = df[column_name].str.replace('\u00C2\u00A0', '', regex=False)

    # Striping punctuation and spliting multi-city entries
    # Take the first city mentioned to maintain a consistent primary location
    df[column_name] = df[column_name].str.strip(' ,.').str.split(r'\s*[/,&]\s*|\s+[Aa]nd\s+', n=1).str[0]
    df[column_name] = df[column_name].str.strip()

    # Canonical Mapping: Fixing typos and consolidating regions
    city_map = {
        'Bangalore': 'Bengaluru', 'Gurugram': 'Gurgaon',
        'Delhi': 'New Delhi', 'Nw Delhi': 'New Delhi',
        'Ahemadabad': 'Ahmedabad', 'Ahemdabad': 'Ahmedabad',
        'Bhubneswar': 'Bhubaneswar', 'Kolkatta': 'Kolkata',
        'SFO': 'San Francisco', 'NY': 'New York',
        'USA': 'United States', 'US': 'United States',
        'Kormangala': 'Bengaluru', 'Andheri': 'Mumbai',
        'Chembur': 'Mumbai', 'Taramani': 'Chennai'
    }
    df[column_name] = df[column_name].replace(city_map)

    # Outlier Re-classification
    # Categorizing state/country level data as 'Not Specified' for city-level analysis
    non_cities = [
        'Karnataka', 'Haryana', 'India', 'California', 'Uttar Pradesh',
        'Kerala', 'United States', 'Missourie', 'Global'
    ]
    df.loc[df[column_name].isin(non_cities), column_name] = 'Not Specified'

    return df

# Executing City Cleaning
df = clean_city_data(df)

def clean_investment_type(df, column_name="Investment Type"):
    """
    Standardizes funding stages into broad categories.
    This reduces categorical cardinality, allowing for more robust
    statistical analysis of the startup lifecycle.
    """
    # Standardizing text format to handle typos and spacing
    df[column_name] = df[column_name].fillna('nan').astype(str).str.lower().str.strip()

    # Cleaning common noise and typos found in the raw dataset
    df[column_name] = df[column_name].str.replace(r'\\\\n', '', regex=True)
    df[column_name] = df[column_name].str.replace(' / ', '/')
    df[column_name] = df[column_name].str.replace('angle', 'angel')
    df[column_name] = df[column_name].str.replace('pre-series', 'pre series')

    # Comprehensive mapping of sub-categories to parent funding stages
    investment_map = {
        'seed': 'Seed Funding', 'seed round': 'Seed Funding',
        'seed funding': 'Seed Funding', 'seed funding round': 'Seed Funding',
        'seedfunding': 'Seed Funding',

        'angel': 'Angel Funding', 'angel round': 'Angel Funding',
        'angel funding': 'Angel Funding',

        'seed/angel funding': 'Seed/Angel Funding',
        'seed/ angel funding': 'Seed/Angel Funding',
        'angel / seed funding': 'Seed/Angel Funding',

        'pre series a': 'Pre-Series A', 'series a': 'Series A',
        'series b': 'Series B', 'series b (extension)': 'Series B',
        'series c': 'Series C', 'series d': 'Series D', 'series e': 'Series E',
        'series f': 'Series F', 'series g': 'Series G', 'series h': 'Series H',
        'series j': 'Series J',

        'venture': 'Venture Capital', 'venture round': 'Venture Capital',
        'single venture': 'Venture Capital', 'venture - series unknown': 'Venture Capital',

        'private equity round': 'Private Equity', 'private equity': 'Private Equity',
        'privatefunding': 'Private Equity', 'private': 'Private Equity',
        'privateequity': 'Private Equity',

        'debt funding': 'Debt Funding', 'debt': 'Debt Funding',
        'debt-funding': 'Debt Funding', 'structured debt': 'Debt Funding',
        'term loan': 'Debt Funding', 'debt and preference capital': 'Debt Funding',

        'crowd funding': 'Crowd Funding', 'corporate round': 'Corporate Round',
        'bridge round': 'Bridge Round',

        'nan': 'Other/Unknown', 'funding round': 'Other/Unknown',
        'maiden round': 'Other/Unknown', 'inhouse funding': 'Other/Unknown',
        'equity': 'Other/Unknown', 'equity based funding': 'Other/Unknown',
        'mezzanine': 'Other/Unknown',
    }

    # Applying mapping and handling outliers
    df[column_name] = df[column_name].map(investment_map).fillna('Other/Unknown')
    return df

# Executing standardizing
df = clean_investment_type(df)

import re
import unicodedata
import pandas as pd
import numpy as np

def _clean_individual_name(name):
    """
    Sub-routine to standardize individual investor names by removing
    legal suffixes and handling missing values.
    """
    if not isinstance(name, str) or pd.isna(name):
        return None

    # Normalizing whitespace and removing trailing punctuation
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'[.,]$', '', name)

    # Stripping legal entity suffixes (Ltd, LLC, Inc, etc.) to consolidate names
    name = re.sub(r'\s+\b(Pte Ltd|Ltd|LLC|Inc|LP|Corp|GmbH|NV|SA)\b', '', name, flags=re.IGNORECASE)
    name = name.strip(' ",.\\/—–_')

    if not name or name.lower() in ['and', 'or', 'others', 'other', '']:
        return None

    # Standardizing 'Not Specified', 'Undisclosed', and 'Angel/Individual' categories
    if re.fullmatch(r'^\s*([Nn]/[Aa]|NaN)\s*$', name, flags=re.IGNORECASE):
        return 'Not Specified'
    if re.search(r'\b(Undisclosed|Unnamed)\b', name, flags=re.IGNORECASE):
        return 'Undisclosed'
    if re.search(r'\b(Angel Investor|Individual Investor|High Net-Worth Individuals|HNI)\b', name, flags=re.IGNORECASE):
        return 'Angel/Individual Investor'

    return name

def clean_investor_names(df, column_name='Investors Name'):
    """
    Standardizes the investor column by decoding byte sequences,
    normalizing Unicode, and exploding multi-investor strings for granular cleaning.
    """
    s = df[column_name].copy().fillna('__NAN__').astype(str)

    def pre_clean_entry(text):
        if text == '__NAN__': return None

        # Resolving legacy encoding issues (Latin-1 to UTF-8 conversion)
        try:
            temp = text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            temp = text

        # Normalizing Unicode characters (NFKC) to handle various keyboard inputs
        temp = unicodedata.normalize('NFKC', temp)

        # Mapping specific escaped byte sequences to proper characters
        replacements = {
            '\\xe2\\x80\\x99': "'", '\\xc2\\xa0': " ", '\\xc3\\x98': 'Ø',
            '\\xc3\\xaf': 'ï', '\\n': ' ', '\\xe2\\x80\\x9d': '"',
            '\\xe2\\x80\\x9c': '"', "\\'": "'", '\\"': '"'
        }
        for old, new in replacements.items():
            temp = temp.replace(old, new)

        # Removing parenthetical info and extra whitespace
        temp = re.sub(r'\s+', ' ', temp).strip()
        temp = re.sub(r'\s*\([^)]+\)', '', temp)
        return temp

    # Pre-cleaning and handling delimiters
    s_pre_cleaned = s.apply(pre_clean_entry)
    s_delimited = s_pre_cleaned.str.replace(r'\s*,\s*|\s+\b(and)\b\s+', '|', regex=True, flags=re.IGNORECASE)

    # Exploding multi-investor entries to clean names individually
    s_exploded = s_delimited.str.split('|').explode()
    s_cleaned = s_exploded.apply(_clean_individual_name)

    # Re-aggregating unique, cleaned names back into the original row structure
    def aggregate_names(names):
        # Filter out None values before processing
        valid_names = [name for name in names if name is not None]
        unique_names = list(dict.fromkeys(valid_names))
        return ', '.join(unique_names) if unique_names else 'Not Specified'

    # Group by the index of s_cleaned (which maintains the original row structure after explode)
    s_final = s_cleaned.groupby(s_cleaned.index).apply(aggregate_names)
    df[column_name] = s_final.reindex(df.index, fill_value='Not Specified')

    return df

# Executing standardizing
df = clean_investor_names(df)

import pandas as pd
import re

# Canonical Mappings for Industry Standardization
# This mapping consolidates fragmented sub-sectors into high-level verticals
INDUSTRY_CANON = {
    "consumer_internet": ["consumer internet", "ecommerce", "e-commerce", "fashion", "retail", "digital media"],
    "technology": ["technology", "tech", "saas", "software", "ai", "artificial intelligence", "data", "cloud", "iot"],
    "healthcare": ["healthcare", "health", "med", "medical", "healthtech", "pharmaceutical"],
    "finance": ["finance", "fintech", "financial", "payment", "banking", "insurance"],
    "logistics": ["logistics", "transport", "supply chain", "delivery", "automobile", "travel"],
    "education": ["education", "ed-tech", "edtech", "elearning"],
    "food": ["food & beverage", "food", "foodtech", "restaurant", "hospitality"],
    "real_estate": ["real estate", "proptech", "property"],
    "gaming": ["gaming", "game"],
    "hrtech": ["hrtech", "hr", "recruitment"],
    "social": ["social", "dating", "community", "networking"]
}

def clean_industry(x):
    """Standardizes industry strings using regex and canonical mapping."""
    if pd.isna(x): return "other"
    s_raw = str(x).strip().lower()
    if s_raw in {"nan", "na", "none", "n/a", "ni", "not applicable", ""}:
        return "other"

    # Robust text normalization
    s_norm = re.sub(r"[\\W_]+", " ", s_raw).strip()

    for canon, variants in INDUSTRY_CANON.items():
        for v in variants:
            if v in s_norm:
                return canon

    # Fallback to the first meaningful word
    s_norm_words = s_norm.split()
    return s_norm_words[0] if s_norm_words and s_norm_words[0] not in {"nan", "na"} else "other"

def infer_industry_from_name(name):
    """
    Implements a heuristic-based inference engine.
    Infers industry from startup names using domain-specific keyword anchors.
    """
    KEYWORD_MAP = {
        'food': ['food', 'zomato', 'swiggy', 'freshmenu', 'box8', 'chef', 'restaurant'],
        'finance': ['pay', 'bank', 'charge', 'bill', 'cash', 'fin', 'money', 'invest', 'razorpay', 'paytm'],
        'education': ['edu', 'learn', 'byju', 'teach', 'academy', 'class', 'simplilearn'],
        'healthcare': ['health', 'med', 'fit', 'practo', 'gym', 'doc', 'pharm', 'clinic'],
        'consumer_internet': ['shop', 'store', 'kart', 'lens', 'print', 'retail', 'ecommerce', 'flipkart', 'amazon'],
        'logistics': ['trip', 'travel', 'rout', 'stay', 'hotel', 'oyo', 'cab', 'delivery', 'transport'],
        'real_estate': ['nest', 'broker', 'house', 'home', 'property', 'flat', 'housing'],
        'technology': ['saas', 'software', 'data', 'cloud', 'tech', 'app', 'ai', 'iot'],
        'hrtech': ['job', 'rozgar', 'hr', 'recruit', 'staffing', 'talent', 'hire'],
        'social': ['roposo', 'chat', 'social', 'community', 'dating', 'crowdfire']
    }

    if pd.isna(name): return None
    name_norm = re.sub(r"[\W_]", "", str(name).strip().lower())

    for industry, keywords in KEYWORD_MAP.items():
        if any(k in name_norm for k in keywords):
            return industry
    return None

def clean_industry_vertical_data(df):
    """
    Main pipeline for industry cleaning. Combines standardization with
    predictive inference to minimize data loss.
    """
    df = df.copy()

    # Tracking originally empty rows to evaluate imputation performance
    orig_empty_mask = df['Industry Vertical'].isna() | \
                      df['Industry Vertical'].str.strip().str.lower().isin(["nan", "na", "none", ""])

    # Standardizing existing data
    df['Industry Vertical'] = df['Industry Vertical'].apply(clean_industry)

    # Heuristic Imputation for 'other' or missing categories
    mask = (df['Industry Vertical'] == "other") & orig_empty_mask
    inferred = df.loc[mask, 'Startup Name'].apply(infer_industry_from_name)
    df.loc[mask, 'Industry Vertical'] = inferred.fillna("other")

    # Final Cleanup & Pruning
    df['Industry Vertical'] = df['Industry Vertical'].apply(clean_industry)
    delete_mask = orig_empty_mask & (df['Industry Vertical'] == "other")

    print(f"Inferred {inferred.notna().sum()} industries. Pruned {delete_mask.sum()} unresolvable records.")
    return df[~delete_mask].copy()

# Executing Industry Pipeline
df = clean_industry_vertical_data(df)


# ==========================================
# SECTION 2: IMPUTATION LOGIC
# ==========================================

# Visualization of Data Sparsity
# The missingno matrix helps identify if missing values are correlated across columns
plt.figure(figsize=(10, 6))
msno.matrix(df)
plt.title("Data Sparsity Matrix: Identifying Missingness Patterns", fontsize=16)
plt.show()

# Target Encoding with Out-of-Fold (OOF) Smoothing
def _target_encode_oof(series, y, n_splits=5, alpha=10.0, random_state=42):
    """
    Implements m-estimate smoothing to encode categorical variables.
    Prevents leakage by using K-Fold Out-of-Fold (OOF) encoding.
    """
    s = series.fillna("Unknown").astype(str)
    y = pd.Series(y).astype(float).values
    global_mean = float(np.mean(y))
    enc = np.empty(len(s), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(s.values):
        s_tr, y_tr = s.values[train_idx], y[train_idx]
        df_tr = pd.DataFrame({"cat": s_tr, "y": y_tr})
        grp = df_tr.groupby("cat")["y"].agg(["sum", "count"])

        # Calculating smoothed mean for the category
        smoothed = (grp["sum"] + alpha * global_mean) / (grp["count"] + alpha)
        enc_fold = pd.Series(s.values[val_idx]).map(smoothed).fillna(global_mean)
        enc[val_idx] = enc_fold.values

    return pd.Series(enc, index=series.index)

# Logistic Regression for Missingness Testing
def logistic_regression_missingness_test(df, var_with_missing, categorical_columns):
    """
    Quantifies if the missingness of a variable (MAR/MNAR) is statistically
    dependent on other categorical features using Logistic Regression.
    """
    df_test = df.copy()
    df_test["is_missing"] = df_test[var_with_missing].isna().astype(int)

    encoded_cols = []
    for col in categorical_columns:
        feat_name = f"{col}_encoded"
        df_test[feat_name] = _target_encode_oof(df_test[col], df_test["is_missing"])
        encoded_cols.append(feat_name)

    # Building the Design Matrix (X)
    X = sm.add_constant(df_test[encoded_cols])
    y = df_test["is_missing"]

    # Fitting Logistic Regression to check for significant predictors
    try:
        model = sm.Logit(y, X).fit(disp=False)
        print("--- Statistical Missingness Summary ---")
        print(model.summary())
        return model
    except Exception as e:
        print(f"Optimization failed: {e}. Data may be perfectly separable.")
        return None

# Executing Test for 'Amount in USD'
categorical_predictors = ["Industry Vertical", "City Location", "Investment Type"]
missingness_model = logistic_regression_missingness_test(df, "Amount in USD", categorical_predictors)

# Configuration for High-Performance Imputation
RANDOM_STATE = 42
MIN_SAMPLES = 30
N_ESTIMATORS = 300
IMPORTANCE_DROP_THRESHOLD = 0.04

def build_base_df(df):
    """
    Normalizes funding stages and extracts temporal features for modeling.
    This function is now designed to accept an already cleaned dataframe.
    """
    df_round = df.copy()
    replace_map = {
        'Seed/ Angel Funding': 'Seed Funding', 'Seed / Angel Funding': 'Seed Funding',
        'Seed/Angel Funding': 'Seed Funding', 'Seed & Angel': 'Seed Funding'
    }
    df_round['Investment Type'] = df_round['Investment Type'].replace(replace_map).fillna('Unknown')
    df_round['Year'] = df_round['Date'].dt.year.fillna(0).astype(int)
    df_round['Month'] = df_round['Date'].dt.month.fillna(0).astype(int)
    return df_round

def compute_investor_mean_cols(df_group, target_col):
    """
    Calculates smoothed mean funding per investor to use as a predictive feature.
    """
    df_known = df_group[df_group[target_col].notna()].copy()
    df_miss  = df_group[df_group[target_col].isna()].copy()

    investor_to_amounts = {}
    for _, row in df_known.iterrows():
        invs = [x.strip() for x in str(row['Investors Name']).split(',') if x.strip()]
        for inv in invs:
            investor_to_amounts.setdefault(inv, []).append(row[target_col])

    investor_mean_map = {inv: np.mean(vals) for inv, vals in investor_to_amounts.items()}
    global_mean = df_known[target_col].mean()

    def compute(cell):
        invs = [x.strip() for x in str(cell).split(',') if x.strip()]
        if not invs: return global_mean
        vals = [investor_mean_map.get(inv) for inv in invs if inv in investor_mean_map]
        return np.mean(vals) if vals else global_mean

    df_known['investor_mean'] = df_known['Investors Name'].apply(compute)
    df_miss['investor_mean']  = df_miss['Investors Name'].apply(compute)
    return df_known, df_miss

def train_per_type_models(df_round, feat_cols, target='Amount in USD'):
    """
    Trains specialized ExtraTrees models for each investment stage.
    Uses log-transformation to stabilize variance in heavy-tailed funding data.
    """
    results_summary = []
    models_by_type = {}

    for inv_type in df_round['Investment Type'].unique():
        df_sub = df_round[df_round['Investment Type'] == inv_type].copy()
        if df_sub[target].notna().sum() < MIN_SAMPLES:
            df_round.loc[df_sub.index, 'Amount_Predicted'] = df_sub[target].median()
            continue

        df_known, df_miss = compute_investor_mean_cols(df_sub, target)

        # Log-transformation handles the extreme outliers typical in Venture Capital
        X = df_known[list(feat_cols)].fillna(0)
        y_log = np.log1p(df_known[target])

        X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=RANDOM_STATE)

        model = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        if len(df_miss) > 0:
            df_round.loc[df_miss.index, 'Amount_Predicted'] = np.expm1(model.predict(df_miss[list(feat_cols)].fillna(0)))

        models_by_type[inv_type] = {'model': model, 'features': list(feat_cols)}

    return df_round, models_by_type

# Probabilistic Record Linkage for Deduplication
def perform_deduplication(df):
    """
    Uses Jaro-Winkler string similarity to identify duplicate startup entries
    that differ only by typos or minor naming conventions.
    """
    indexer = recordlinkage.Index()
    indexer.block('City Location') # Blocking reduces computational complexity O(n^2)
    pairs = indexer.index(df)

    compare = recordlinkage.Compare()
    compare.string('Startup Name', 'Startup Name', method='jarowinkler', threshold=0.85, label='startup_name')
    compare.string('Investors Name', 'Investors Name', method='jarowinkler', threshold=0.85, label='investors_name')
    compare.exact('Industry Vertical', 'Industry Vertical', label='industry')

    features = compare.compute(pairs, df)
    potential_duplicates = features[features.mean(axis=1) > 0.85]
    return potential_duplicates

# Building the training set & running the Imputation Engine
df_prepared = build_base_df(df)
# Use these features because Baseline test showed they are most important
features_to_use = ['Year', 'Month', 'investor_mean']

df_final, models = train_per_type_models(df_prepared, features_to_use)

# Creating the finalized amount column (Actuals + Imputed)
df_final['Amount_Final'] = df_final['Amount in USD'].fillna(df_final['Amount_Predicted'])

# Probabilistic Deduplication
duplicates = perform_deduplication(df_final)
print(f"Analysis Complete. Found {len(duplicates)} duplicate records.")

# Saving the result so the Visualization script can use it
df_final.to_csv('startup_funding_cleaned_and_imputed.csv', index=False)


# ==========================================
# SECTION 3: VISUALIZATION FUNCTIONS
# ==========================================

# ------PLOT 1--------
investor_list = []
for x in df['Investors Name'].dropna():
    investor_list.extend([i.strip() for i in str(x).split(',') if i.strip()])

top_inv = pd.Series(investor_list).value_counts().head(15).reset_index()
top_inv.columns = ['Investor', 'Count']

fig = px.bar(
    top_inv.sort_values('Count', ascending=True), 
    x='Count', y='Investor', 
    orientation='h',
    color='Count',
    color_continuous_scale='sunset_r', 
    title="<b>Market Liquidity: Most Active Investors (Deal Count)</b>"
)

fig.update_layout(
    template="plotly_white", 
    showlegend=False,
    xaxis_title="Number of Deals",
    yaxis_title="Investor Name",
    title_font=dict(size=22, family="Arial", color="black"),
    xaxis=dict(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey'),
    yaxis=dict(showline=True, linewidth=2, linecolor='black'),
    font=dict(color="black", size=14)
)

fig.update_coloraxes(cmin=top_inv['Count'].min() - 5, showscale=False)

fig.update_traces(
    marker_line_width=1, 
    marker_line_color="black",
    opacity=1.0
)

fig.show()

# ------PLOT 2--------
pivot_df = df_final.pivot_table(
    index='Industry Vertical',
    columns='Year',
    values='Amount_Final',
    aggfunc='count'
).fillna(0)

top_15_names = df_final.groupby('Industry Vertical')['Amount_Final'].sum().nlargest(15).index
pivot_df = pivot_df.loc[top_15_names]

fig = px.imshow(
    pivot_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale=[[0, '#FF69B4'], [0.5, '#C71585'], [1, '#800000']],
    labels=dict(x="Year", y="Industry", color="Deal Count"),
    title="<b>Sector Maturity Heatmap: Deal Frequency (2015-2020)</b>"
)

fig.update_layout(
    template="plotly_white",
    title_font=dict(size=22, family="Arial", color="black"),
    font=dict(color="black", size=12),
    xaxis=dict(side="bottom", tickfont=dict(weight='bold')),
    yaxis=dict(tickfont=dict(weight='bold'))
)

fig.update_traces(
    xgap=3,
    ygap=3,
    textfont=dict(size=14, family="Arial Black")
)

fig.show()

# ------PLOT 3 --------
def calc_resilience(df_slice):
    sector_year = df_slice.groupby(['Industry Vertical', 'Year'])['Amount_Final'].sum().reset_index()
    stats = sector_year.groupby('Industry Vertical')['Amount_Final'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats[stats['count'] >= 2]
    stats['Resilience Index'] = 1 - (stats['std'] / stats['mean'].replace(0, np.nan))
    return stats.sort_values('Resilience Index', ascending=False)
res_data = calc_resilience(df_final)
fig_res = px.bar(
    res_data.head(15),
    x='Resilience Index',
    y='Industry Vertical',
    orientation='h',
    color='Resilience Index',
    color_continuous_scale='Plasma',
    title="<b>Top 15 Resilient Sectors (Funding Stability Score)</b>"
)

fig_res.update_layout(
    template="plotly_white",
    yaxis={'categoryorder':'total ascending'},
    xaxis_title="Resilience Index (1 - CV)",
    font=dict(color="black")
)
fig_res.show()

# ------PLOT 4 --------
trend = df_final.groupby('Year')['Amount_Final'].sum().reset_index()

fig_trend = px.line(
    trend, x='Year', y='Amount_Final', markers=True,
    title="<b>Aggregate Funding Trend: Longitudinal Analysis (USD - Imputed)</b>",
)

fig_trend.update_layout(
    template="plotly_white",
    xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
    yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True, title="Total Funding (USD)"),
    font=dict(family="Arial", size=14, color="black"),
    title_font=dict(size=22)
)

fig_trend.update_traces(
    line=dict(color='#8B0000', width=4),
    marker=dict(size=12, color='#FF1493', line=dict(width=2, color='black'))
)

fig_trend.show()
