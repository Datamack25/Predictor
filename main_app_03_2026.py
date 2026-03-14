import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
import pytz
import time
import json
from fpdf import FPDF
import io
import base64
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import re
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderSIA
    _vader = _VaderSIA()
    VADER_AVAILABLE = True
except ImportError:
    _vader = None
    VADER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #0e1117; }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #1e3a5f;
        text-align: center;
    }
    .main-header h1 { 
        color: #00d4ff; 
        font-size: 2.2em; 
        margin: 0;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
    }
    .main-header p { color: #8892b0; margin: 5px 0 0 0; font-size: 0.95em; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1f2e, #16213e);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-label { color: #8892b0; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ccd6f6; font-size: 1.6em; font-weight: bold; margin: 5px 0; }
    .metric-delta-up { color: #64ffda; font-size: 0.9em; }
    .metric-delta-down { color: #ff6b6b; font-size: 0.9em; }
    
    /* Sentiment gauge */
    .sentiment-container {
        background: linear-gradient(145deg, #1a1f2e, #16213e);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Signal badges */
    .signal-bull {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        color: #6ee7b7;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .signal-bear {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border: 1px solid #3b82f6;
        color: #93c5fd;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Section titles */
    .section-title {
        color: #00d4ff;
        font-size: 1.1em;
        font-weight: bold;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 8px;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* News item */
    .news-item {
        background: #1a1f2e;
        border-left: 3px solid #00d4ff;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .news-title { color: #ccd6f6; font-size: 0.9em; }
    .news-meta { color: #8892b0; font-size: 0.75em; margin-top: 4px; }
    .news-positive { border-left-color: #10b981; }
    .news-negative { border-left-color: #ef4444; }
    .news-neutral { border-left-color: #3b82f6; }
    
    /* Probability bars */
    .prob-container { margin: 10px 0; }
    .prob-label { color: #8892b0; font-size: 0.85em; margin-bottom: 4px; }
    
    /* Source badges */
    .source-badge {
        background: #1e3a5f;
        color: #93c5fd;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        display: inline-block;
        margin: 2px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1f2e; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }
    
    /* Timestamp */
    .update-time {
        color: #8892b0;
        font-size: 0.8em;
        text-align: right;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MARKETS = {
    "NASDAQ 100": {"ticker": "^NDX", "icon": "🇺🇸", "color": "#00d4ff"},
    "S&P 500":    {"ticker": "^GSPC", "icon": "🇺🇸", "color": "#64ffda"},
    "Or (Gold)":  {"ticker": "GC=F",  "icon": "🥇",  "color": "#ffd700"},
    "CAC 40":     {"ticker": "^FCHI", "icon": "🇫🇷",  "color": "#a78bfa"},
    "Euronext 600": {"ticker": "^STOXX", "icon": "🇪🇺", "color": "#f472b6"},
    "Pétrole (WTI)": {"ticker": "CL=F", "icon": "🛢️", "color": "#fb923c"},
}

TIMEFRAMES = {
    "5 min":   {"minutes": 5,   "yf_period": "1d",  "yf_interval": "5m"},
    "15 min":  {"minutes": 15,  "yf_period": "5d",  "yf_interval": "15m"},
    "60 min":  {"minutes": 60,  "yf_period": "1mo", "yf_interval": "60m"},
    "240 min": {"minutes": 240, "yf_period": "3mo", "yf_interval": "1h"},
}

RSS_FEEDS = {
    "Reuters Finance":     "https://feeds.reuters.com/reuters/businessNews",
    "Bloomberg Markets":   "https://feeds.bloomberg.com/markets/news.rss",
    "FT Markets":          "https://www.ft.com/markets?format=rss",
    "Investing.com":       "https://www.investing.com/rss/news.rss",
    "Yahoo Finance":       "https://finance.yahoo.com/news/rssindex",
    "MarketWatch":         "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC":                "https://feeds.nbcnews.com/nbcnews/public/business",
    "Les Echos":           "https://feeds.lesechos.fr/lesechos-finance",
    "Le Monde Eco":        "https://www.lemonde.fr/economie/rss_full.xml",
    "ECB News":            "https://www.ecb.europa.eu/rss/press.html",
}

# ─────────────────────────────────────────────────────────────
# TWITTER/X KEY ACCOUNTS  (scraped via Nitter public RSS)
# Categories: Finance · Macro · Géopolitique · Banques centrales · Tech
# ─────────────────────────────────────────────────────────────
TWITTER_ACCOUNTS = {
    # ── Macro & Économie ──
    "Elon Musk":            {"handle": "elonmusk",        "cat": "Tech/Géopol",   "weight": 0.9},
    "Nouriel Roubini":      {"handle": "nouriel",         "cat": "Macro Bear",    "weight": 0.85},
    "Ray Dalio":            {"handle": "raydalio",        "cat": "Macro",         "weight": 0.90},
    "Mohamed El-Erian":     {"handle": "mohamedelerian",  "cat": "Macro/BCE",     "weight": 0.88},
    "Larry Summers":        {"handle": "lawrencehsummers","cat": "Macro Policy",  "weight": 0.85},
    "Cathie Wood":          {"handle": "cathiedwood",     "cat": "Innovation",    "weight": 0.80},
    "Michael Burry":        {"handle": "michaeljburry",   "cat": "Macro Bear",    "weight": 0.87},
    "Bill Ackman":          {"handle": "billackman",      "cat": "Macro/Equity",  "weight": 0.85},
    "Stanley Druckenmiller":{"handle": "stan_druckenmiller","cat": "Macro",       "weight": 0.90},
    # ── Finance & Trading ──
    "Jim Cramer":           {"handle": "jimcramer",       "cat": "Finance",       "weight": 0.70},
    "Carl Icahn":           {"handle": "carlicahn",       "cat": "Activist",      "weight": 0.80},
    "Scott Bessent":        {"handle": "scottbessent",    "cat": "Macro/Trésor",  "weight": 0.85},
    "David Einhorn":        {"handle": "davideinhorn",    "cat": "Value",         "weight": 0.80},
    "Paul Tudor Jones":     {"handle": "ptj_official",    "cat": "Macro Trading", "weight": 0.88},
    "Jeff Gundlach":        {"handle": "trancool",        "cat": "Bonds/Macro",   "weight": 0.85},
    # ── Banques centrales & Politique ──
    "Fed Reserve":          {"handle": "federalreserve",  "cat": "Banque centrale","weight": 0.95},
    "ECB":                  {"handle": "ecb",             "cat": "Banque centrale","weight": 0.95},
    "IMF":                  {"handle": "imf",             "cat": "Institution",   "weight": 0.90},
    "World Bank":           {"handle": "worldbank",       "cat": "Institution",   "weight": 0.85},
    "BIS":                  {"handle": "bis_org",         "cat": "Banque centrale","weight": 0.90},
    # ── Géopolitique ──
    "Ian Bremmer":          {"handle": "ianbremmer",      "cat": "Géopolitique",  "weight": 0.88},
    "George Friedman":      {"handle": "georgefriedman",  "cat": "Géopolitique",  "weight": 0.85},
    "Niall Ferguson":        {"handle": "nfergus",         "cat": "Histoire/Géo",  "weight": 0.82},
    "Fareed Zakaria":       {"handle": "fareedzakaria",   "cat": "Géopolitique",  "weight": 0.82},
    "Francis Fukuyama":     {"handle": "fukuyama_francis","cat": "Géopolitique",  "weight": 0.80},
    # ── Crypto & Tech Finance ──
    "Michael Saylor":       {"handle": "saylor",          "cat": "Bitcoin/Macro", "weight": 0.80},
    "Balaji Srinivasan":    {"handle": "balajis",         "cat": "Tech/Macro",    "weight": 0.78},
    "Raoul Pal":            {"handle": "raoulpal",        "cat": "Macro/Crypto",  "weight": 0.82},
    # ── Media Finance FR ──
    "BFM Bourse":           {"handle": "bfmbusiness",     "cat": "Finance FR",    "weight": 0.75},
    "Les Echos":            {"handle": "lesechos",        "cat": "Finance FR",    "weight": 0.78},
}

# Nitter instances (public, no auth required) — tried in order
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.mint.lgbt",
    "https://nitter.unixfox.eu",
]

TWITTER_CATEGORIES = sorted(set(v["cat"] for v in TWITTER_ACCOUNTS.values()))

ECONOMIC_INDICATORS = {
    "Inflation US (CPI)":       {"value": 3.2,  "trend": "↘", "impact": "négatif", "weight": 0.12},
    "Taux Fed (FFR)":           {"value": 5.25, "trend": "→", "impact": "négatif", "weight": 0.15},
    "Taux BCE":                 {"value": 4.50, "trend": "→", "impact": "négatif", "weight": 0.10},
    "Inflation Zone Euro":      {"value": 2.6,  "trend": "↘", "impact": "neutre",  "weight": 0.08},
    "PMI Manufacturier US":     {"value": 52.1, "trend": "↗", "impact": "positif", "weight": 0.09},
    "PMI Services US":          {"value": 54.3, "trend": "↗", "impact": "positif", "weight": 0.09},
    "Chômage US":               {"value": 3.9,  "trend": "→", "impact": "neutre",  "weight": 0.07},
    "PIB US (QoQ)":             {"value": 2.8,  "trend": "↗", "impact": "positif", "weight": 0.10},
    "Indice Dollar (DXY)":      {"value": 104.2,"trend": "↗", "impact": "négatif", "weight": 0.08},
    "Courbe des taux (2-10y)":  {"value": -0.35,"trend": "↗", "impact": "négatif", "weight": 0.07},
    "VIX (Peur marché)":        {"value": 16.8, "trend": "↘", "impact": "positif", "weight": 0.10},
    "Balance commerciale US":   {"value": -67.4,"trend": "↘", "impact": "neutre",  "weight": 0.05},
    "Confiance consommateur":   {"value": 102.0,"trend": "↗", "impact": "positif", "weight": 0.08},
    "ISM Manufacturing":        {"value": 49.2, "trend": "↗", "impact": "négatif", "weight": 0.07},
    "Tension géopolitique":     {"value": 6.5,  "trend": "↗", "impact": "négatif", "weight": 0.09},
    "Sentiment IA/Tech":        {"value": 72.0, "trend": "↗", "impact": "positif", "weight": 0.10},
    "Élections/Risque politique":{"value": 5.5, "trend": "↗", "impact": "négatif", "weight": 0.06},
    "Flux institutionnels":     {"value": 68.0, "trend": "↗", "impact": "positif", "weight": 0.08},
    "Earnings S&P500 Growth":   {"value": 8.2,  "trend": "↗", "impact": "positif", "weight": 0.09},
    "Liquidité Fed (M2)":       {"value": 20800,"trend": "↗", "impact": "positif", "weight": 0.07},
}


# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_ohlc(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with automatic fallback."""
    fallback_chain = [
        (period, interval),
        ("5d",  "5m"),
        ("1mo", "15m"),
        ("3mo", "1h"),
        ("6mo", "1d"),
    ]
    for p, i in fallback_chain:
        try:
            t  = yf.Ticker(ticker)
            df = t.history(period=p, interval=i)
            if df.empty or len(df) < 10:
                continue
            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 10:
                return df
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_news_sentiment() -> list:
    """Fetch and analyze news from RSS feeds."""
    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                published = entry.get("published", "")
                if not title:
                    continue
                
                # Sentiment analysis
                blob = TextBlob(title + " " + summary[:200])
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                sentiment_label = "neutre"
                if polarity > 0.05:
                    sentiment_label = "positif"
                elif polarity < -0.05:
                    sentiment_label = "négatif"
                
                articles.append({
                    "source": source,
                    "title": title[:120],
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "sentiment": sentiment_label,
                    "published": published[:30] if published else "",
                    "link": entry.get("link", "#"),
                })
        except Exception:
            continue
    
    return articles


@st.cache_data(ttl=900)
def fetch_twitter_sentiment(max_per_account: int = 5) -> list:
    """
    Fetch recent tweets from key accounts via Nitter RSS (no API key needed).
    Falls back gracefully if all instances are down.
    """
    articles = []

    for account_name, meta in TWITTER_ACCOUNTS.items():
        handle   = meta["handle"]
        cat      = meta["cat"]
        weight   = meta["weight"]
        fetched  = False

        for nitter_base in NITTER_INSTANCES:
            rss_url = f"{nitter_base}/{handle}/rss"
            try:
                feed = feedparser.parse(rss_url)
                if not feed.entries:
                    continue

                for entry in feed.entries[:max_per_account]:
                    title   = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    pub     = entry.get("published", "")

                    # Strip HTML tags from summary
                    if summary:
                        try:
                            from bs4 import BeautifulSoup as _BS
                            summary = _BS(summary, "html.parser").get_text()[:300]
                        except Exception:
                            summary = summary[:300]

                    if not title or len(title) < 5:
                        continue

                    # NLP sentiment
                    blob = TextBlob(title + " " + summary[:200])
                    polarity      = blob.sentiment.polarity
                    subjectivity  = blob.sentiment.subjectivity

                    # Weight-adjusted polarity
                    adj_polarity  = polarity * weight

                    sentiment_label = "neutre"
                    if polarity > 0.05:  sentiment_label = "positif"
                    elif polarity < -0.05: sentiment_label = "négatif"

                    articles.append({
                        "source":       f"@{handle}",
                        "account_name": account_name,
                        "category":     cat,
                        "title":        title[:150],
                        "polarity":     polarity,
                        "adj_polarity": adj_polarity,
                        "subjectivity": subjectivity,
                        "sentiment":    sentiment_label,
                        "published":    pub[:30] if pub else "",
                        "weight":       weight,
                        "link":         entry.get("link", "#"),
                        "type":         "twitter",
                    })

                fetched = True
                break  # got data from this instance, move to next account

            except Exception:
                continue  # try next nitter instance

        if not fetched:
            # Account unreachable — add placeholder so UI shows source
            articles.append({
                "source": f"@{handle}", "account_name": account_name,
                "category": cat, "title": f"[Données non disponibles — {account_name}]",
                "polarity": 0.0, "adj_polarity": 0.0, "subjectivity": 0.0,
                "sentiment": "neutre", "published": "", "weight": weight,
                "link": f"https://twitter.com/{handle}", "type": "twitter_unavailable",
            })

    return articles


def compute_twitter_score(twitter_articles: list) -> dict:
    """Compute aggregate sentiment score from Twitter accounts."""
    available = [a for a in twitter_articles if a.get("type") != "twitter_unavailable"]
    if not available:
        return {"score": 50, "bull": 0, "bear": 0, "neutral": 0,
                "weighted_polarity": 0.0, "by_category": {}}

    bull    = sum(1 for a in available if a["sentiment"] == "positif")
    bear    = sum(1 for a in available if a["sentiment"] == "négatif")
    neutral = len(available) - bull - bear

    # Weighted average polarity
    total_w  = sum(a["weight"] for a in available)
    wtd_pol  = sum(a["adj_polarity"] for a in available) / max(total_w, 1)

    # Score 0-100
    score = 50 + wtd_pol * 200  # ±0.25 polarity → ±50 pts
    score = float(np.clip(score, 5, 95))

    # By category
    by_cat = {}
    for a in available:
        cat = a["category"]
        if cat not in by_cat:
            by_cat[cat] = {"bull": 0, "bear": 0, "neutral": 0, "polarity": []}
        if a["sentiment"] == "positif":   by_cat[cat]["bull"] += 1
        elif a["sentiment"] == "négatif": by_cat[cat]["bear"] += 1
        else:                             by_cat[cat]["neutral"] += 1
        by_cat[cat]["polarity"].append(a["polarity"])

    for cat in by_cat:
        pols = by_cat[cat]["polarity"]
        by_cat[cat]["avg_polarity"] = round(float(np.mean(pols)), 3) if pols else 0.0

    return {
        "score":             round(score, 1),
        "bull":              bull,
        "bear":              bear,
        "neutral":           neutral,
        "total":             len(available),
        "weighted_polarity": round(wtd_pol, 4),
        "by_category":       by_cat,
    }


@st.cache_data(ttl=120)
def get_current_price(ticker: str) -> dict:
    """Get real-time price info."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        hist = t.history(period="2d", interval="1m")
        if hist.empty:
            return {}
        
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
        change = last - prev
        change_pct = (change / prev) * 100 if prev != 0 else 0
        
        return {
            "price": last,
            "change": change,
            "change_pct": change_pct,
            "high": hist["High"].max(),
            "low": hist["Low"].min(),
            "volume": hist["Volume"].sum(),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators."""
    if df.empty or len(df) < 20:
        return df
    
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]
    
    # Moving averages
    df["EMA9"]  = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_mid"]   = sma20
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    
    # Volume SMA
    df["Vol_SMA"] = vol.rolling(20).mean()
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + vol.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - vol.iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    
    return df


# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
def compute_technical_signal(df: pd.DataFrame) -> dict:
    """Generate buy/sell signals from technical indicators."""
    if df.empty or len(df) < 30:
        return {"score": 50, "signals": [], "direction": "NEUTRE"}
    
    df = compute_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = []
    bull_score = 0
    bear_score = 0
    total_weight = 0
    
    # RSI
    if not np.isnan(last.get("RSI", np.nan)):
        rsi = last["RSI"]
        if rsi < 30:
            signals.append({"name": "RSI oversold", "value": f"{rsi:.1f}", "signal": "🟢 BULL", "weight": 8})
            bull_score += 8
        elif rsi > 70:
            signals.append({"name": "RSI overbought", "value": f"{rsi:.1f}", "signal": "🔴 BEAR", "weight": 8})
            bear_score += 8
        elif 40 < rsi < 60:
            signals.append({"name": "RSI neutre", "value": f"{rsi:.1f}", "signal": "⚪ NEUTRE", "weight": 4})
        total_weight += 8
    
    # MACD
    if not np.isnan(last.get("MACD", np.nan)):
        macd = last["MACD"]
        macd_sig = last["MACD_signal"]
        macd_prev = prev["MACD"]
        sig_prev  = prev["MACD_signal"]
        if macd > macd_sig and macd_prev <= sig_prev:
            signals.append({"name": "MACD crossover ↑", "value": f"{macd:.4f}", "signal": "🟢 BULL", "weight": 10})
            bull_score += 10
        elif macd < macd_sig and macd_prev >= sig_prev:
            signals.append({"name": "MACD crossover ↓", "value": f"{macd:.4f}", "signal": "🔴 BEAR", "weight": 10})
            bear_score += 10
        elif macd > macd_sig:
            signals.append({"name": "MACD positif", "value": f"{macd:.4f}", "signal": "🟢 BULL", "weight": 6})
            bull_score += 6
        else:
            signals.append({"name": "MACD négatif", "value": f"{macd:.4f}", "signal": "🔴 BEAR", "weight": 6})
            bear_score += 6
        total_weight += 10
    
    # EMA Cross
    if not np.isnan(last.get("EMA9", np.nan)) and not np.isnan(last.get("EMA21", np.nan)):
        ema9  = last["EMA9"]
        ema21 = last["EMA21"]
        close = last["Close"]
        if ema9 > ema21 and close > ema9:
            signals.append({"name": "EMA9 > EMA21", "value": f"{ema9:.2f}", "signal": "🟢 BULL", "weight": 9})
            bull_score += 9
        elif ema9 < ema21 and close < ema9:
            signals.append({"name": "EMA9 < EMA21", "value": f"{ema9:.2f}", "signal": "🔴 BEAR", "weight": 9})
            bear_score += 9
        total_weight += 9
    
    # Bollinger Bands
    if not np.isnan(last.get("BB_upper", np.nan)):
        close = last["Close"]
        bb_up = last["BB_upper"]
        bb_lo = last["BB_lower"]
        bb_mid= last["BB_mid"]
        if close > bb_up:
            signals.append({"name": "BB: prix > upper", "value": f"{close:.2f}", "signal": "🔴 BEAR", "weight": 7})
            bear_score += 7
        elif close < bb_lo:
            signals.append({"name": "BB: prix < lower", "value": f"{close:.2f}", "signal": "🟢 BULL", "weight": 7})
            bull_score += 7
        elif close > bb_mid:
            signals.append({"name": "BB: prix > mid", "value": f"{close:.2f}", "signal": "🟢 BULL", "weight": 4})
            bull_score += 4
        total_weight += 7
    
    # Stochastic
    if not np.isnan(last.get("Stoch_K", np.nan)):
        k = last["Stoch_K"]
        d = last["Stoch_D"]
        if k < 20 and k > d:
            signals.append({"name": "Stoch: rebond", "value": f"{k:.1f}", "signal": "🟢 BULL", "weight": 7})
            bull_score += 7
        elif k > 80 and k < d:
            signals.append({"name": "Stoch: retournement", "value": f"{k:.1f}", "signal": "🔴 BEAR", "weight": 7})
            bear_score += 7
        total_weight += 7
    
    # Volume confirmation
    if "Vol_SMA" in df.columns and not np.isnan(last.get("Vol_SMA", np.nan)):
        vol = last["Volume"]
        vol_sma = last["Vol_SMA"]
        close = last["Close"]
        prev_close = prev["Close"]
        if vol > vol_sma * 1.5 and close > prev_close:
            signals.append({"name": "Volume achat ↑↑", "value": f"x{vol/vol_sma:.1f}", "signal": "🟢 BULL", "weight": 8})
            bull_score += 8
        elif vol > vol_sma * 1.5 and close < prev_close:
            signals.append({"name": "Volume vente ↑↑", "value": f"x{vol/vol_sma:.1f}", "signal": "🔴 BEAR", "weight": 8})
            bear_score += 8
        total_weight += 8
    
    # Price momentum
    if len(df) >= 5:
        ret5 = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100
        if ret5 > 0.5:
            signals.append({"name": "Momentum +5 bars", "value": f"+{ret5:.2f}%", "signal": "🟢 BULL", "weight": 6})
            bull_score += 6
        elif ret5 < -0.5:
            signals.append({"name": "Momentum -5 bars", "value": f"{ret5:.2f}%", "signal": "🔴 BEAR", "weight": 6})
            bear_score += 6
        total_weight += 6
    
    total = bull_score + bear_score
    if total == 0:
        score = 50
    else:
        score = (bull_score / total) * 100
    
    if score >= 65:
        direction = "HAUSSIER"
    elif score <= 35:
        direction = "BAISSIER"
    else:
        direction = "NEUTRE"
    
    return {
        "score": score,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "direction": direction,
        "signals": signals,
    }


def compute_fundamental_score(news_articles: list, twitter_score: dict = None) -> dict:
    """Compute fundamental/macro sentiment score including Twitter signals."""
    eco_bull = 0
    eco_bear = 0
    eco_signals = []

    for name, data in ECONOMIC_INDICATORS.items():
        w = data["weight"] * 10
        impact = data["impact"]
        if impact == "positif":
            eco_bull += w
            eco_signals.append({"name": name, "signal": "🟢", "value": str(data["value"])})
        elif impact == "négatif":
            eco_bear += w
            eco_signals.append({"name": name, "signal": "🔴", "value": str(data["value"])})
        else:
            eco_signals.append({"name": name, "signal": "⚪", "value": str(data["value"])})

    # News sentiment score (RSS)
    news_bull  = sum(1 for a in news_articles if a["sentiment"] == "positif")
    news_bear  = sum(1 for a in news_articles if a["sentiment"] == "négatif")
    news_total = max(len(news_articles), 1)
    news_score = (news_bull / news_total) * 100

    # Twitter/X sentiment score
    tw_score = twitter_score.get("score", 50) if twitter_score else 50

    # Combined: eco 60% · news RSS 20% · Twitter 20%
    eco_total = eco_bull + eco_bear
    eco_score = (eco_bull / eco_total * 100) if eco_total > 0 else 50
    combined  = eco_score * 0.60 + news_score * 0.20 + tw_score * 0.20

    return {
        "eco_score":    eco_score,
        "news_score":   news_score,
        "twitter_score": tw_score,
        "combined":     combined,
        "eco_signals":  eco_signals,
        "news_bull":    news_bull,
        "news_bear":    news_bear,
        "news_neutral": news_total - news_bull - news_bear,
    }


def predict_direction(tech_score: float, fund_score: float, timeframe_minutes: int) -> dict:
    """Predict direction with probabilities per timeframe."""
    # Weight: short-term = more technical, long-term = more fundamental
    if timeframe_minutes <= 15:
        w_tech, w_fund = 0.80, 0.20
    elif timeframe_minutes <= 60:
        w_tech, w_fund = 0.65, 0.35
    else:
        w_tech, w_fund = 0.50, 0.50
    
    combined = tech_score * w_tech + fund_score * w_fund
    
    # Add slight random noise for realism
    noise = np.random.normal(0, 2)
    combined = np.clip(combined + noise, 5, 95)
    
    bull_prob  = combined / 100
    bear_prob  = (100 - combined) / 100
    
    # Confidence based on distance from 50%
    confidence = abs(combined - 50) / 50 * 100
    
    if combined >= 60:
        direction = "HAUSSIER"
        color = "#10b981"
        emoji = "📈"
    elif combined <= 40:
        direction = "BAISSIER"
        color = "#ef4444"
        emoji = "📉"
    else:
        direction = "NEUTRE"
        color = "#3b82f6"
        emoji = "➡️"
    
    return {
        "direction": direction,
        "score": combined,
        "bull_prob": round(bull_prob * 100, 1),
        "bear_prob": round(bear_prob * 100, 1),
        "confidence": round(confidence, 1),
        "color": color,
        "emoji": emoji,
        "w_tech": w_tech,
        "w_fund": w_fund,
    }


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
def make_candlestick_chart(df: pd.DataFrame, market_name: str, color: str) -> go.Figure:
    """Create a full OHLC chart with indicators."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Données non disponibles")
        return fig
    
    df = compute_indicators(df)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=("OHLC + Bollinger + EMA", "Volume", "RSI", "MACD")
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="OHLC",
        increasing_fillcolor="#10b981", increasing_line_color="#10b981",
        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
    ), row=1, col=1)
    
    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(255,200,0,0.4)", dash="dash", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(255,200,0,0.4)", dash="dash", width=1),
            fill="tonexty", fillcolor="rgba(255,200,0,0.05)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid",
            line=dict(color="rgba(255,200,0,0.6)", width=1)), row=1, col=1)
    
    # EMAs
    if "EMA9" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9",
            line=dict(color="#a78bfa", width=1.5)), row=1, col=1)
    if "EMA21" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA21",
            line=dict(color="#f472b6", width=1.5)), row=1, col=1)
    
    # Volume
    colors_vol = ["#10b981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef4444"
                  for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors_vol, showlegend=False), row=2, col=1)
    if "Vol_SMA" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Vol_SMA"], name="Vol SMA",
            line=dict(color="#fbbf24", width=1)), row=2, col=1)
    
    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#00d4ff", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot",  line_color="gray",    row=3, col=1)
    
    # MACD
    if "MACD" in df.columns:
        macd_colors = ["#10b981" if v >= 0 else "#ef4444" for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
            marker_color=macd_colors, showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#00d4ff", width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color="#f472b6", width=1.5)), row=4, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(text=f"📊 {market_name} — Analyse Technique", font=dict(color="#00d4ff", size=16)),
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#8892b0"),
        margin=dict(t=80, b=20),
    )
    
    for i in range(1, 5):
        fig.update_yaxes(gridcolor="#1e3a5f", row=i, col=1)
        fig.update_xaxes(gridcolor="#1e3a5f", row=i, col=1)
    
    return fig


def make_probability_chart(predictions: dict) -> go.Figure:
    """Create probability radar/bar chart for all timeframes."""
    timeframes = list(predictions.keys())
    bull_probs = [predictions[tf]["bull_prob"] for tf in timeframes]
    bear_probs = [predictions[tf]["bear_prob"] for tf in timeframes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Hausse 📈",
        x=timeframes,
        y=bull_probs,
        marker_color=["#10b981" if p >= 50 else "#6ee7b7" for p in bull_probs],
        text=[f"{p}%" for p in bull_probs],
        textposition="inside",
        textfont=dict(size=14, color="white"),
    ))
    
    fig.add_trace(go.Bar(
        name="Baisse 📉",
        x=timeframes,
        y=bear_probs,
        marker_color=["#ef4444" if p >= 50 else "#fca5a5" for p in bear_probs],
        text=[f"{p}%" for p in bear_probs],
        textposition="inside",
        textfont=dict(size=14, color="white"),
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.3)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        barmode="group",
        title=dict(text="Probabilités par horizon temporel", font=dict(color="#00d4ff")),
        xaxis_title="Horizon",
        yaxis_title="Probabilité (%)",
        yaxis=dict(range=[0, 100]),
        height=350,
        font=dict(color="#8892b0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def make_sentiment_gauge(score: float, title: str) -> go.Figure:
    """Create a sentiment gauge."""
    if score >= 60:
        bar_color = "#10b981"
    elif score <= 40:
        bar_color = "#ef4444"
    else:
        bar_color = "#3b82f6"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 50, "valueformat": ".1f"},
        number={"suffix": "%", "font": {"color": bar_color, "size": 28}},
        title={"text": title, "font": {"color": "#ccd6f6", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8892b0", "tickfont": {"size": 10}},
            "bar": {"color": bar_color, "thickness": 0.8},
            "bgcolor": "#1a1f2e",
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [0, 35],  "color": "rgba(239,68,68,0.15)"},
                {"range": [35, 65], "color": "rgba(59,130,246,0.15)"},
                {"range": [65, 100],"color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    
    fig.update_layout(
        paper_bgcolor="#0e1117",
        font=dict(color="#8892b0"),
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


def make_eco_indicators_chart() -> go.Figure:
    """Heatmap of economic indicators."""
    names = list(ECONOMIC_INDICATORS.keys())
    values = []
    colors = []
    
    for n, d in ECONOMIC_INDICATORS.items():
        w = d["weight"] * 10
        if d["impact"] == "positif":
            values.append(w)
            colors.append(w)
        elif d["impact"] == "négatif":
            values.append(-w)
            colors.append(-w)
        else:
            values.append(0)
            colors.append(0)
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(
            color=colors,
            colorscale=[[0, "#ef4444"], [0.5, "#3b82f6"], [1, "#10b981"]],
            cmin=-max(abs(v) for v in values),
            cmax=max(abs(v) for v in values),
        ),
        text=[f"{ECONOMIC_INDICATORS[n]['value']} ({ECONOMIC_INDICATORS[n]['trend']})" for n in names],
        textposition="inside",
        textfont=dict(size=10, color="white"),
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(text="📊 Indicateurs Économiques & Macro", font=dict(color="#00d4ff")),
        height=550,
        font=dict(color="#8892b0"),
        xaxis_title="Impact pondéré",
        margin=dict(l=220, r=20, t=50, b=20),
    )
    fig.update_yaxes(tickfont=dict(size=10))
    return fig



# ─────────────────────────────────────────────────────────────
# HELPER: sanitize text for PDF (remove non-latin1 chars)
# ─────────────────────────────────────────────────────────────
def _p(text: str) -> str:
    """Remove characters not supported by FPDF Helvetica (latin-1 only)."""
    # Replace common unicode arrows and special chars
    replacements = {
        '↗': '>>', '↘': '<<', '→': '->', '←': '<-',
        '↑': '^',  '↓': 'v',  '•': '-',  '–': '-',
        '—': '-',  '…': '...','é': 'e',  'è': 'e',
        'ê': 'e',  'ë': 'e',  'à': 'a',  'â': 'a',
        'ù': 'u',  'û': 'u',  'î': 'i',  'ï': 'i',
        'ô': 'o',  'ç': 'c',  'É': 'E',  'È': 'E',
        'Ê': 'E',  'À': 'A',  'Â': 'A',  'Î': 'I',
        'Ô': 'O',  'Û': 'U',  'Ç': 'C',  '°': 'deg',
        '²': '2',  '³': '3',  '½': '1/2','¼': '1/4',
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Final pass: encode to latin-1, replacing anything else
    return text.encode('latin-1', errors='replace').decode('latin-1')


# ─────────────────────────────────────────────────────────────
# SCALPING CONSEILS ENGINE
# ─────────────────────────────────────────────────────────────
def compute_scalp_advice(df: pd.DataFrame, pred: dict, market_name: str, timeframe: str, trade_size: float = 100.0) -> dict:
    """Compute entry/exit levels and trade advice for scalpers."""
    if df.empty:
        return {}

    # Work with whatever rows we have, minimum 5
    if len(df) < 5:
        return {}

    # If fewer than 20 rows, build a synthetic price-based fallback
    if len(df) < 20:
        current_price = df["Close"].iloc[-1]
        atr = (df["High"] - df["Low"]).mean()
        if np.isnan(atr) or atr == 0:
            atr = current_price * 0.001
        direction  = pred.get("direction", "NEUTRE")
        bull_prob  = pred.get("bull_prob", 50)
        bear_prob  = pred.get("bear_prob", 50)
        confidence = pred.get("confidence", 0)
        rsi = 50.0
        bb_upper = current_price * 1.005
        bb_lower = current_price * 0.995
        ema9  = current_price
        ema21 = current_price
    else:
        df = compute_indicators(df)
        last = df.iloc[-1]

        current_price = last["Close"]
        atr = last.get("ATR", current_price * 0.001)
        if np.isnan(atr) or atr == 0:
            atr = current_price * 0.001

        direction  = pred.get("direction", "NEUTRE")
        bull_prob  = pred.get("bull_prob", 50)
        bear_prob  = pred.get("bear_prob", 50)
        confidence = pred.get("confidence", 0)

        # ATR multipliers by timeframe

    # ATR multipliers by timeframe
    tf_mult = {"5 min": 0.8, "15 min": 1.2, "60 min": 2.0, "240 min": 3.5}
    mult = tf_mult.get(timeframe, 1.0)
    mult = tf_mult.get(timeframe, 1.0)

    bb_upper = last.get("BB_upper", current_price * 1.005)
    bb_lower = last.get("BB_lower", current_price * 0.995)
    ema9     = last.get("EMA9",     current_price)
    ema21    = last.get("EMA21",    current_price)
    rsi      = last.get("RSI",      50)

    if np.isnan(bb_upper): bb_upper = current_price * 1.005
    if np.isnan(bb_lower): bb_lower = current_price * 0.995
    if np.isnan(ema9):     ema9     = current_price
    if np.isnan(ema21):    ema21    = current_price
    if np.isnan(rsi):      rsi      = 50

    # ── LONG setup ──
    if direction == "HAUSSIER":
        entry_long   = round(current_price * 1.0001, 4)          # légèrement au-dessus du bid
        sl_long      = round(current_price - atr * mult * 1.2, 4)
        tp1_long     = round(current_price + atr * mult * 1.5, 4)
        tp2_long     = round(current_price + atr * mult * 2.5, 4)
        tp3_long     = round(min(bb_upper, current_price + atr * mult * 4), 4)
        rr_long      = round((tp1_long - entry_long) / max(entry_long - sl_long, 0.0001), 2)

        entry_short  = round(current_price * 0.9999, 4)
        sl_short     = round(current_price + atr * mult * 2.0, 4)  # SL serré contre-trend
        tp1_short    = round(current_price - atr * mult * 0.8, 4)
        tp2_short    = round(current_price - atr * mult * 1.5, 4)
        tp3_short    = None
        rr_short     = round((entry_short - tp1_short) / max(sl_short - entry_short, 0.0001), 2)

    elif direction == "BAISSIER":
        entry_short  = round(current_price * 0.9999, 4)
        sl_short     = round(current_price + atr * mult * 1.2, 4)
        tp1_short    = round(current_price - atr * mult * 1.5, 4)
        tp2_short    = round(current_price - atr * mult * 2.5, 4)
        tp3_short    = round(max(bb_lower, current_price - atr * mult * 4), 4)
        rr_short     = round((entry_short - tp1_short) / max(sl_short - entry_short, 0.0001), 2)

        entry_long   = round(current_price * 1.0001, 4)
        sl_long      = round(current_price - atr * mult * 2.0, 4)
        tp1_long     = round(current_price + atr * mult * 0.8, 4)
        tp2_long     = round(current_price + atr * mult * 1.5, 4)
        tp3_long     = None
        rr_long      = round((tp1_long - entry_long) / max(entry_long - sl_long, 0.0001), 2)

    else:  # NEUTRE
        entry_long   = round(bb_lower * 1.001, 4)
        sl_long      = round(bb_lower * 0.998, 4)
        tp1_long     = round(current_price, 4)
        tp2_long     = round(bb_upper * 0.999, 4)
        tp3_long     = None
        rr_long      = round((tp1_long - entry_long) / max(entry_long - sl_long, 0.0001), 2)

        entry_short  = round(bb_upper * 0.999, 4)
        sl_short     = round(bb_upper * 1.002, 4)
        tp1_short    = round(current_price, 4)
        tp2_short    = round(bb_lower * 1.001, 4)
        tp3_short    = None
        rr_short     = round((entry_short - tp1_short) / max(sl_short - entry_short, 0.0001), 2)

    # ── P&L sur 100$ ──
    def pnl_calc(entry, tp1, sl, size=100.0):
        if entry == 0:
            return 0, 0
        gain_pct = abs(tp1 - entry) / entry * 100
        loss_pct = abs(sl  - entry) / entry * 100
        gain_usd = round(size * gain_pct / 100, 2)
        loss_usd = round(size * loss_pct / 100, 2)
        return gain_usd, loss_usd

    gain_long,  loss_long  = pnl_calc(entry_long,  tp1_long,  sl_long,  trade_size)
    gain_short, loss_short = pnl_calc(entry_short, tp1_short, sl_short, trade_size)

    # ── Probabilité de gain ──
    # basée sur bull_prob + RSI + confidence
    rsi_bias_long  = (rsi - 50) / 50  # positif si RSI > 50
    prob_gain_long  = min(95, max(5, bull_prob + rsi_bias_long * 5 + confidence * 0.1))
    prob_gain_short = min(95, max(5, bear_prob - rsi_bias_long * 5 + confidence * 0.1))

    # ── Résumé conseil ──
    if direction == "HAUSSIER" and confidence >= 40:
        main_advice = "LONG PRIORITAIRE"
        advice_color = "#10b981"
        advice_icon = "🟢"
    elif direction == "BAISSIER" and confidence >= 40:
        main_advice = "SHORT PRIORITAIRE"
        advice_color = "#ef4444"
        advice_icon = "🔴"
    elif rsi < 30:
        main_advice = "LONG (RSI oversold)"
        advice_color = "#10b981"
        advice_icon = "🟢"
    elif rsi > 70:
        main_advice = "SHORT (RSI overbought)"
        advice_color = "#ef4444"
        advice_icon = "🔴"
    else:
        main_advice = "ATTENDRE SIGNAL CLAIR"
        advice_color = "#3b82f6"
        advice_icon = "⚪"

    return {
        "current_price": current_price,
        "atr": atr,
        "rsi": rsi,
        "direction": direction,
        "bull_prob": bull_prob,
        "bear_prob": bear_prob,
        "confidence": confidence,
        "main_advice": main_advice,
        "advice_color": advice_color,
        "advice_icon": advice_icon,
        # Long
        "entry_long":  entry_long,
        "sl_long":     sl_long,
        "tp1_long":    tp1_long,
        "tp2_long":    tp2_long,
        "tp3_long":    tp3_long,
        "rr_long":     rr_long,
        "gain_long":   gain_long,
        "loss_long":   loss_long,
        "prob_gain_long": round(prob_gain_long, 1),
        # Short
        "entry_short": entry_short,
        "sl_short":    sl_short,
        "tp1_short":   tp1_short,
        "tp2_short":   tp2_short,
        "tp3_short":   tp3_short,
        "rr_short":    rr_short,
        "gain_short":  gain_short,
        "loss_short":  loss_short,
        "prob_gain_short": round(prob_gain_short, 1),
        # Levels
        "bb_upper": round(bb_upper, 4),
        "bb_lower": round(bb_lower, 4),
        "ema9":     round(ema9, 4),
        "ema21":    round(ema21, 4),
    }


# ─────────────────────────────────────────────────────────────
# PDF REPORT  (fixed: all text sanitized via _p())
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(
    market_data: dict,
    predictions_all: dict,
    fund_score: dict,
    news_articles: list = None,
    scalp_advices: dict = None,
) -> bytes:
    """
    Compact 2-3 page PDF:
      Page 1 — Resume executif + Conseils scalping tous horizons
      Page 2 — Indicateurs techniques par marche
      Page 3 — Actualites & sentiment news
    All text sanitized via _p() for latin-1 compatibility.
    """
    if news_articles is None:
        news_articles = []
    if scalp_advices is None:
        scalp_advices = {}

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    # ── helpers ──────────────────────────────────────────
    def title_bar(text, r=30, g=100, b=180):
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _p(text), ln=True, fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(1)

    def section_bar(text):
        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(20, 60, 130)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, _p(f"  {text}"), ln=True, fill=True)
        pdf.set_text_color(30, 30, 30)

    def row(label, value, r=30, g=30, b=30, bold_val=False):
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(70, 5, _p(label), border=0)
        pdf.set_font("Helvetica", "B" if bold_val else "", 8)
        pdf.set_text_color(r, g, b)
        pdf.cell(0, 5, _p(str(value)), ln=True)
        pdf.set_text_color(30, 30, 30)

    def divider():
        pdf.set_draw_color(200, 210, 230)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ════════════════════════════════════════════════════
    # PAGE 1 — RESUME EXECUTIF + CONSEILS SCALPING
    # ════════════════════════════════════════════════════
    pdf.add_page()

    # Header
    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _p("MARKET SENTIMENT PREDICTOR"), ln=True, fill=True, align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_fill_color(30, 80, 150)
    pdf.cell(0, 5, _p(f"Rapport genere le {now_str}  |  NASDAQ  S&P500  Or  CAC40  Euronext600  Petrole"), ln=True, fill=True, align="C")
    pdf.ln(3)

    # ── Resume executif ──
    title_bar("1. RESUME EXECUTIF — SCORES GLOBAUX")

    # Scores en colonnes
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(240, 245, 255)
    pdf.set_text_color(20, 60, 130)
    for label, val in [
        ("Score Fondamental Global",  f"{fund_score['combined']:.1f} / 100"),
        ("Score Macro-Economique",    f"{fund_score['eco_score']:.1f} / 100"),
        ("Score Sentiment News",      f"{fund_score['news_score']:.1f} / 100"),
        ("Actualites analysees",      f"{fund_score['news_bull']+fund_score['news_bear']+fund_score['news_neutral']} articles  |  +{fund_score['news_bull']} positives  -{fund_score['news_bear']} negatives  ={fund_score['news_neutral']} neutres"),
    ]:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(75, 5, _p(label))
        pdf.set_font("Helvetica", "B", 8)
        # Color based on value
        if "Score" in label:
            v = float(str(val).split("/")[0].strip())
            if v >= 60:   pdf.set_text_color(0, 140, 80)
            elif v <= 40: pdf.set_text_color(200, 40, 40)
            else:         pdf.set_text_color(30, 100, 200)
        else:
            pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 5, _p(val), ln=True)

    pdf.ln(2)
    divider()

    # ── Previsions marches (tableau compact) ──
    title_bar("2. PREVISIONS PAR MARCHE & HORIZON")

    # Header tableau
    pdf.set_fill_color(200, 215, 240)
    pdf.set_text_color(20, 60, 130)
    pdf.set_font("Helvetica", "B", 7)
    pdf.cell(35, 5, "Marche",   fill=True, border=1)
    pdf.cell(18, 5, "Prix",     fill=True, border=1)
    pdf.cell(10, 5, "Var%",     fill=True, border=1)
    for tf in ["5 min", "15 min", "60 min", "240 min"]:
        pdf.cell(32, 5, tf, fill=True, border=1, align="C")
    pdf.ln()

    for mkt_name, mkt_data in market_data.items():
        pi = mkt_data.get("price_info", {})
        price = pi.get("price", 0)
        chg   = pi.get("change_pct", 0)
        try:
            price_str = f"{price:,.2f}"
            chg_str   = f"{chg:+.2f}%"
        except Exception:
            price_str = "N/A"
            chg_str   = "N/A"

        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(30, 30, 30)
        pdf.set_fill_color(250, 251, 255)
        pdf.cell(35, 5, _p(mkt_name[:18]), fill=True, border=1)

        pdf.set_font("Helvetica", "", 7)
        pdf.cell(18, 5, _p(price_str), fill=True, border=1, align="R")

        if chg >= 0: pdf.set_text_color(0, 130, 70)
        else:        pdf.set_text_color(200, 40, 40)
        pdf.cell(10, 5, _p(chg_str), fill=True, border=1, align="C")
        pdf.set_text_color(30, 30, 30)

        preds = predictions_all.get(mkt_name, {})
        for tf in ["5 min", "15 min", "60 min", "240 min"]:
            p = preds.get(tf, {})
            d = p.get("direction", "—")
            b = p.get("bull_prob", 50)
            cell_txt = f"{d[:4]} {b}%"
            if d == "HAUSSIER":   pdf.set_text_color(0, 130, 70);  pdf.set_fill_color(240, 255, 245)
            elif d == "BAISSIER": pdf.set_text_color(200, 40, 40); pdf.set_fill_color(255, 242, 242)
            else:                 pdf.set_text_color(30, 80, 180);  pdf.set_fill_color(242, 246, 255)
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(32, 5, _p(cell_txt), fill=True, border=1, align="C")
        pdf.set_text_color(30, 30, 30)
        pdf.ln()

    pdf.ln(2)
    divider()

    # ── Conseils scalping tous horizons ──
    title_bar("3. CONSEILS SCALPING — TOUS HORIZONS", r=0, g=100, b=80)

    for mkt_name, tf_advices in scalp_advices.items():
        if not tf_advices:
            continue
        section_bar(f">> {mkt_name}")

        # Mini header
        pdf.set_fill_color(230, 245, 240)
        pdf.set_text_color(20, 80, 60)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(20, 4, "Horizon",    fill=True, border=1)
        pdf.cell(15, 4, "Conseil",    fill=True, border=1)
        pdf.cell(22, 4, "Entry Long", fill=True, border=1, align="C")
        pdf.cell(22, 4, "SL Long",    fill=True, border=1, align="C")
        pdf.cell(22, 4, "TP1 Long",   fill=True, border=1, align="C")
        pdf.cell(15, 4, "R:R",        fill=True, border=1, align="C")
        pdf.cell(18, 4, "ProbGain",   fill=True, border=1, align="C")
        pdf.cell(18, 4, f"G/P $",     fill=True, border=1, align="C")
        pdf.cell(18, 4, "Prix actu",  fill=True, border=1, align="C")
        pdf.ln()

        for tf_name, adv in tf_advices.items():
            if not adv:
                continue
            d = adv.get("direction", "—")
            conseil = adv.get("main_advice", "—")[:14]
            entry_l = adv.get("entry_long", 0)
            sl_l    = adv.get("sl_long", 0)
            tp1_l   = adv.get("tp1_long", 0)
            rr_l    = adv.get("rr_long", 0)
            pg_l    = adv.get("prob_gain_long", 0)
            gain_l  = adv.get("gain_long", 0)
            loss_l  = adv.get("loss_long", 0)
            cp      = adv.get("current_price", 0)

            if d == "HAUSSIER":   pdf.set_text_color(0, 120, 60);  pdf.set_fill_color(245, 255, 248)
            elif d == "BAISSIER": pdf.set_text_color(180, 30, 30); pdf.set_fill_color(255, 245, 245)
            else:                 pdf.set_text_color(40, 80, 170); pdf.set_fill_color(245, 248, 255)

            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(20, 4, _p(tf_name), fill=True, border=1)
            pdf.set_font("Helvetica", "", 7)
            pdf.cell(15, 4, _p(conseil), fill=True, border=1)
            pdf.set_text_color(0, 120, 60)
            pdf.cell(22, 4, _p(f"{entry_l:,.4f}"), fill=True, border=1, align="R")
            pdf.set_text_color(180, 30, 30)
            pdf.cell(22, 4, _p(f"{sl_l:,.4f}"),    fill=True, border=1, align="R")
            pdf.set_text_color(200, 130, 0)
            pdf.cell(22, 4, _p(f"{tp1_l:,.4f}"),   fill=True, border=1, align="R")
            # RR color
            if rr_l >= 1.5:   pdf.set_text_color(0, 120, 60)
            elif rr_l >= 1.0: pdf.set_text_color(200, 130, 0)
            else:             pdf.set_text_color(180, 30, 30)
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(15, 4, _p(f"1:{rr_l}"), fill=True, border=1, align="C")
            # Prob gain
            if pg_l >= 60:   pdf.set_text_color(0, 120, 60)
            elif pg_l >= 50: pdf.set_text_color(200, 130, 0)
            else:            pdf.set_text_color(180, 30, 30)
            pdf.cell(18, 4, _p(f"{pg_l}%"), fill=True, border=1, align="C")
            # G/P
            pdf.set_text_color(30, 30, 30)
            pdf.set_font("Helvetica", "", 7)
            pdf.cell(18, 4, _p(f"+{gain_l:.2f}/-{loss_l:.2f}"), fill=True, border=1, align="C")
            pdf.cell(18, 4, _p(f"{cp:,.3f}"), fill=True, border=1, align="R")
            pdf.ln()

        pdf.ln(1)

    # ════════════════════════════════════════════════════
    # PAGE 2 — INDICATEURS TECHNIQUES
    # ════════════════════════════════════════════════════
    pdf.add_page()

    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _p("INDICATEURS TECHNIQUES PAR MARCHE"), ln=True, fill=True, align="C")
    pdf.set_font("Helvetica", "", 7)
    pdf.set_fill_color(30, 80, 150)
    pdf.cell(0, 4, _p(f"Genere le {now_str}"), ln=True, fill=True, align="C")
    pdf.ln(3)

    TECH_INDICATORS = [
        ("RSI (14)",       "RSI",        lambda v: f"{v:.1f}",   lambda v: ("SURBUY" if v>70 else "SURVENTE" if v<30 else "NEUTRE"), lambda v: (0,130,70) if v<30 else (180,30,30) if v>70 else (40,80,170)),
        ("EMA 9",          "EMA9",       lambda v: f"{v:,.3f}",  lambda v: "—", lambda v: (30,30,30)),
        ("EMA 21",         "EMA21",      lambda v: f"{v:,.3f}",  lambda v: "—", lambda v: (30,30,30)),
        ("MACD",           "MACD",       lambda v: f"{v:.4f}",   lambda v: ("BULL" if v>0 else "BEAR"), lambda v: (0,130,70) if v>0 else (180,30,30)),
        ("BB Upper",       "BB_upper",   lambda v: f"{v:,.3f}",  lambda v: "—", lambda v: (30,30,30)),
        ("BB Lower",       "BB_lower",   lambda v: f"{v:,.3f}",  lambda v: "—", lambda v: (30,30,30)),
        ("ATR (14)",       "ATR",        lambda v: f"{v:.4f}",   lambda v: "—", lambda v: (30,30,30)),
        ("Stoch K",        "Stoch_K",    lambda v: f"{v:.1f}",   lambda v: ("OVERBOUGHT" if v>80 else "OVERSOLD" if v<20 else "NEUTRE"), lambda v: (180,30,30) if v>80 else (0,130,70) if v<20 else (30,30,30)),
    ]

    for mkt_name, mkt_data in market_data.items():
        df_mkt = mkt_data.get("df", pd.DataFrame())
        if df_mkt.empty or len(df_mkt) < 15:
            continue

        # Compute indicators
        try:
            df_ind = compute_indicators(df_mkt.copy())
        except Exception:
            continue

        last = df_ind.iloc[-1]
        pi   = mkt_data.get("price_info", {})
        price = pi.get("price", last.get("Close", 0))
        chg   = pi.get("change_pct", 0)

        section_bar(f"{mkt_name}  |  Prix: {price:,.3f}  |  Var: {chg:+.2f}%")

        # Table header
        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(20, 60, 130)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(35, 4, "Indicateur", fill=True, border=1)
        pdf.cell(35, 4, "Valeur",     fill=True, border=1, align="R")
        pdf.cell(30, 4, "Signal",     fill=True, border=1, align="C")
        pdf.ln()

        for label, col, fmt, signal_fn, color_fn in TECH_INDICATORS:
            val = last.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            sig   = signal_fn(val)
            r,g,b = color_fn(val)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(60, 60, 60)
            pdf.set_fill_color(250, 251, 255)
            pdf.cell(35, 4, _p(label), fill=True, border=1)
            pdf.set_text_color(r, g, b)
            pdf.cell(35, 4, _p(fmt(val)), fill=True, border=1, align="R")
            pdf.cell(30, 4, _p(sig),      fill=True, border=1, align="C")
            pdf.ln()

        pdf.ln(2)

    # ════════════════════════════════════════════════════
    # PAGE 3 — ACTUALITES & SENTIMENT NEWS
    # ════════════════════════════════════════════════════
    pdf.add_page()

    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _p("ACTUALITES & SENTIMENT NEWS"), ln=True, fill=True, align="C")
    pdf.set_font("Helvetica", "", 7)
    pdf.set_fill_color(30, 80, 150)
    pdf.cell(0, 4, _p(f"Genere le {now_str}  |  Analyse NLP TextBlob sur flux RSS"), ln=True, fill=True, align="C")
    pdf.ln(3)

    # Stats sentiment
    nb_pos = fund_score["news_bull"]
    nb_neg = fund_score["news_bear"]
    nb_neu = fund_score["news_neutral"]
    total  = nb_pos + nb_neg + nb_neu

    title_bar("RESUME SENTIMENT", r=40, g=40, b=120)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 5, _p(
        f"Total: {total} articles  |  "
        f"Positifs: {nb_pos} ({nb_pos/max(total,1)*100:.0f}%)  |  "
        f"Negatifs: {nb_neg} ({nb_neg/max(total,1)*100:.0f}%)  |  "
        f"Neutres: {nb_neu} ({nb_neu/max(total,1)*100:.0f}%)"
    ), ln=True)
    pdf.ln(2)

    # Table articles
    title_bar("TOP ACTUALITES ANALYSEES", r=40, g=40, b=120)

    pdf.set_fill_color(220, 225, 245)
    pdf.set_text_color(20, 50, 130)
    pdf.set_font("Helvetica", "B", 7)
    pdf.cell(20, 4, "Sentiment", fill=True, border=1, align="C")
    pdf.cell(18, 4, "Polarite",  fill=True, border=1, align="C")
    pdf.cell(42, 4, "Source",    fill=True, border=1)
    pdf.cell(0,  4, "Titre",     fill=True, border=1)
    pdf.ln()

    # Sort: most positive first, then most negative
    sorted_news = sorted(news_articles, key=lambda a: abs(a.get("polarity", 0)), reverse=True)

    for article in sorted_news[:40]:
        sentiment = article.get("sentiment", "neutre")
        polarity  = article.get("polarity", 0)
        source    = article.get("source", "")[:20]
        title     = article.get("title", "")[:75]

        if sentiment == "positif":
            pdf.set_fill_color(242, 255, 247)
            pdf.set_text_color(0, 130, 60)
            sent_txt = "POSITIF"
        elif sentiment == "negatif" or sentiment == "négatif":
            pdf.set_fill_color(255, 242, 242)
            pdf.set_text_color(180, 30, 30)
            sent_txt = "NEGATIF"
        else:
            pdf.set_fill_color(245, 247, 255)
            pdf.set_text_color(60, 80, 160)
            sent_txt = "NEUTRE"

        pdf.set_font("Helvetica", "B", 6)
        pdf.cell(20, 4, _p(sent_txt),              fill=True, border=1, align="C")
        pdf.set_font("Helvetica", "", 6)
        pdf.cell(18, 4, _p(f"{polarity:+.3f}"),    fill=True, border=1, align="C")
        pdf.set_text_color(60, 60, 60)
        pdf.cell(42, 4, _p(source),                fill=True, border=1)
        pdf.cell(0,  4, _p(title),                 fill=True, border=1)
        pdf.ln()

    # Footer avertissement
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4, _p(
        "AVERTISSEMENT: Ce rapport est genere automatiquement a des fins educatives uniquement. "
        "Il ne constitue pas un conseil financier. Consultez un professionnel avant toute decision d'investissement."
    ))

    return bytes(pdf.output())






# ═══════════════════════════════════════════════════════════════════
# MULTI-AGENT SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# MULTI-AGENT SIMULATION ENGINE  (MiroFish-inspired)
# ═══════════════════════════════════════════════════════════════════

def run_agent_macro_strategist(df: pd.DataFrame, fund_score: dict) -> dict:
    """Macro Strategist Agent — reads economic environment."""
    eco = fund_score.get("eco_score", 50)
    vix_proxy = ECONOMIC_INDICATORS.get("VIX (Peur marché)", {}).get("value", 20)
    yield_curve = ECONOMIC_INDICATORS.get("Courbe des taux (2-10y)", {}).get("value", 0)
    fed_rate = ECONOMIC_INDICATORS.get("Taux Fed (FFR)", {}).get("value", 5.25)
    dxy = ECONOMIC_INDICATORS.get("Indice Dollar (DXY)", {}).get("value", 104)

    # Score macro bull/bear
    score = 50.0
    signals = []

    if vix_proxy < 15:
        score += 8; signals.append(("VIX faible → calme de marché", "+8", "#10b981"))
    elif vix_proxy > 25:
        score -= 10; signals.append(("VIX élevé → peur dominante", "-10", "#ef4444"))
    else:
        score -= 2; signals.append(("VIX modéré", "-2", "#3b82f6"))

    if yield_curve > 0:
        score += 6; signals.append(("Courbe normale → croissance", "+6", "#10b981"))
    elif yield_curve < -0.5:
        score -= 8; signals.append(("Courbe inversée → récession", "-8", "#ef4444"))
    else:
        score -= 3; signals.append(("Courbe plate → incertitude", "-3", "#fbbf24"))

    if fed_rate > 5.0:
        score -= 6; signals.append(("Taux élevés → pression sur actions", "-6", "#ef4444"))
    elif fed_rate < 2.0:
        score += 8; signals.append(("Taux bas → liquidité abondante", "+8", "#10b981"))

    if dxy > 105:
        score -= 4; signals.append(("Dollar fort → pression sur commodités", "-4", "#ef4444"))
    elif dxy < 100:
        score += 4; signals.append(("Dollar faible → favorable marchés émergents", "+4", "#10b981"))

    score = np.clip(score + (eco - 50) * 0.3, 5, 95)
    return {"name": "🏛️ Macro Strategist", "score": round(score, 1), "signals": signals,
            "bias": "BULL" if score > 55 else "BEAR" if score < 45 else "NEUTRAL",
            "confidence": round(abs(score - 50) * 2, 1)}


def run_agent_technical_analyst(df: pd.DataFrame) -> dict:
    """Technical Analyst Agent — pure price action."""
    if df.empty or len(df) < 20:
        return {"name": "📐 Technical Analyst", "score": 50.0, "signals": [],
                "bias": "NEUTRAL", "confidence": 0}

    df2 = compute_indicators(df.copy())
    last = df2.iloc[-1]
    prev = df2.iloc[-2] if len(df2) > 1 else last

    score = 50.0
    signals = []

    # RSI
    rsi = last.get("RSI", 50)
    if not np.isnan(rsi):
        if rsi < 30:   score += 12; signals.append((f"RSI={rsi:.0f} → oversold rebond probable", "+12", "#10b981"))
        elif rsi > 70: score -= 12; signals.append((f"RSI={rsi:.0f} → overbought correction probable", "-12", "#ef4444"))
        elif rsi > 55: score += 4;  signals.append((f"RSI={rsi:.0f} → momentum positif", "+4", "#10b981"))
        elif rsi < 45: score -= 4;  signals.append((f"RSI={rsi:.0f} → momentum négatif", "-4", "#ef4444"))

    # MACD cross
    macd = last.get("MACD", 0); msig = last.get("MACD_signal", 0)
    pmacd = prev.get("MACD", 0); pmsig = prev.get("MACD_signal", 0)
    if not np.isnan(macd):
        if macd > msig and pmacd <= pmsig: score += 14; signals.append(("MACD golden cross ↑", "+14", "#10b981"))
        elif macd < msig and pmacd >= pmsig: score -= 14; signals.append(("MACD death cross ↓", "-14", "#ef4444"))
        elif macd > msig: score += 5; signals.append(("MACD positif", "+5", "#10b981"))
        else: score -= 5; signals.append(("MACD négatif", "-5", "#ef4444"))

    # EMA
    ema9 = last.get("EMA9", 0); ema21 = last.get("EMA21", 0); close = last["Close"]
    if not np.isnan(ema9):
        if close > ema9 > ema21: score += 8; signals.append(("Prix > EMA9 > EMA21 → trend haussier", "+8", "#10b981"))
        elif close < ema9 < ema21: score -= 8; signals.append(("Prix < EMA9 < EMA21 → trend baissier", "-8", "#ef4444"))

    # BB position
    bbu = last.get("BB_upper", close*1.01); bbl = last.get("BB_lower", close*0.99)
    if not np.isnan(bbu):
        bb_pct = (close - bbl) / (bbu - bbl + 1e-10) * 100
        if bb_pct > 85:   score -= 6; signals.append((f"Prix en zone haute BB ({bb_pct:.0f}%)", "-6", "#ef4444"))
        elif bb_pct < 15: score += 6; signals.append((f"Prix en zone basse BB ({bb_pct:.0f}%)", "+6", "#10b981"))

    # Momentum 10 bars
    if len(df2) >= 10:
        ret = (df2["Close"].iloc[-1] / df2["Close"].iloc[-10] - 1) * 100
        if ret > 1.5:   score += 7; signals.append((f"Momentum +10 bars: {ret:+.2f}%", "+7", "#10b981"))
        elif ret < -1.5: score -= 7; signals.append((f"Momentum +10 bars: {ret:+.2f}%", "-7", "#ef4444"))

    score = np.clip(score, 5, 95)
    return {"name": "📐 Technical Analyst", "score": round(score, 1), "signals": signals,
            "bias": "BULL" if score > 55 else "BEAR" if score < 45 else "NEUTRAL",
            "confidence": round(abs(score - 50) * 2, 1)}


def run_agent_sentiment_analyst(news_articles: list, fund_score: dict) -> dict:
    """Sentiment Analyst Agent — NLP + social proxy."""
    news_score = fund_score.get("news_score", 50)
    nb_pos = fund_score.get("news_bull", 0)
    nb_neg = fund_score.get("news_bear", 0)
    total  = max(nb_pos + nb_neg + fund_score.get("news_neutral", 1), 1)

    score = 50.0
    signals = []

    # Ratio positif/négatif
    ratio = nb_pos / max(nb_neg, 1)
    if ratio > 2.5:  score += 15; signals.append((f"Ratio pos/neg={ratio:.1f} → euphorie médiatique", "+15", "#10b981"))
    elif ratio > 1.5: score += 8; signals.append((f"Ratio pos/neg={ratio:.1f} → biais positif", "+8", "#10b981"))
    elif ratio < 0.5: score -= 12; signals.append((f"Ratio pos/neg={ratio:.1f} → panique médiatique", "-12", "#ef4444"))
    elif ratio < 0.8: score -= 6; signals.append((f"Ratio pos/neg={ratio:.1f} → biais négatif", "-6", "#ef4444"))
    else:             signals.append((f"Ratio pos/neg={ratio:.1f} → équilibré", "0", "#3b82f6"))

    # Intensité polarité moyenne
    if news_articles:
        avg_pol = np.mean([a.get("polarity", 0) for a in news_articles])
        if avg_pol > 0.15:   score += 10; signals.append((f"Polarité moyenne={avg_pol:+.3f} → fortement positif", "+10", "#10b981"))
        elif avg_pol > 0.05: score += 5;  signals.append((f"Polarité moyenne={avg_pol:+.3f} → légèrement positif", "+5", "#10b981"))
        elif avg_pol < -0.15: score -= 10; signals.append((f"Polarité moyenne={avg_pol:+.3f} → fortement négatif", "-10", "#ef4444"))
        elif avg_pol < -0.05: score -= 5;  signals.append((f"Polarité moyenne={avg_pol:+.3f} → légèrement négatif", "-5", "#ef4444"))

    # Tech/AI sentiment proxy
    tech_sent = ECONOMIC_INDICATORS.get("Sentiment IA/Tech", {}).get("value", 50)
    if tech_sent > 70: score += 6; signals.append((f"Sentiment IA/Tech={tech_sent} -> euphorique", "+6", "#10b981"))
    elif tech_sent < 40: score -= 6; signals.append((f"Sentiment IA/Tech={tech_sent} -> pessimiste", "-6", "#ef4444"))

    # Twitter signal via fund_score if available
    tw_sc = fund_score.get("twitter_score", 50)
    if tw_sc > 60:   score += 8; signals.append((f"Twitter score={tw_sc:.0f} -> signal haussier", "+8", "#10b981"))
    elif tw_sc < 40: score -= 8; signals.append((f"Twitter score={tw_sc:.0f} -> signal baissier", "-8", "#ef4444"))

    score = np.clip(score, 5, 95)
    return {"name": "📱 Sentiment Analyst", "score": round(score, 1), "signals": signals,
            "bias": "BULL" if score > 55 else "BEAR" if score < 45 else "NEUTRAL",
            "confidence": round(abs(score - 50) * 2, 1)}


def run_agent_earnings_analyst(market_name: str, price_info: dict) -> dict:
    """Earnings / Fundamentals Agent — valuation & growth proxy."""
    score = 50.0
    signals = []

    eps_growth = ECONOMIC_INDICATORS.get("Earnings S&P500 Growth", {}).get("value", 0)
    pmi_mfg    = ECONOMIC_INDICATORS.get("PMI Manufacturier US", {}).get("value", 50)
    pmi_svc    = ECONOMIC_INDICATORS.get("PMI Services US", {}).get("value", 50)
    gdp        = ECONOMIC_INDICATORS.get("PIB US (QoQ)", {}).get("value", 2)
    consumer   = ECONOMIC_INDICATORS.get("Confiance consommateur", {}).get("value", 100)

    if eps_growth > 8:   score += 12; signals.append((f"EPS growth={eps_growth}% → fort", "+12", "#10b981"))
    elif eps_growth > 4: score += 6;  signals.append((f"EPS growth={eps_growth}% → correct", "+6", "#10b981"))
    elif eps_growth < 0: score -= 10; signals.append((f"EPS growth={eps_growth}% → contraction", "-10", "#ef4444"))

    if pmi_mfg > 52: score += 5; signals.append((f"PMI Mfg={pmi_mfg} → expansion", "+5", "#10b981"))
    elif pmi_mfg < 48: score -= 5; signals.append((f"PMI Mfg={pmi_mfg} → contraction", "-5", "#ef4444"))

    if pmi_svc > 54: score += 6; signals.append((f"PMI Services={pmi_svc} → solide", "+6", "#10b981"))
    elif pmi_svc < 50: score -= 5; signals.append((f"PMI Services={pmi_svc} → faible", "-5", "#ef4444"))

    if gdp > 2.5: score += 7; signals.append((f"PIB={gdp}% → croissance saine", "+7", "#10b981"))
    elif gdp < 0: score -= 10; signals.append((f"PIB={gdp}% → récession", "-10", "#ef4444"))

    if consumer > 105: score += 5; signals.append((f"Confiance conso={consumer} → élevée", "+5", "#10b981"))
    elif consumer < 90: score -= 5; signals.append((f"Confiance conso={consumer} → faible", "-5", "#ef4444"))

    # Adjust for commodity vs equity
    if market_name in ("Or (Gold)", "Pétrole (WTI)"):
        inflation = ECONOMIC_INDICATORS.get("Inflation US (CPI)", {}).get("value", 3)
        if inflation > 3.5 and market_name == "Or (Gold)":
            score += 8; signals.append((f"Inflation={inflation}% → Or attractif", "+8", "#10b981"))
        if market_name == "Pétrole (WTI)":
            geopo = ECONOMIC_INDICATORS.get("Tension géopolitique", {}).get("value", 5)
            if geopo > 7: score += 8; signals.append((f"Tensions géopo={geopo}/10 → soutien pétrole", "+8", "#10b981"))

    score = np.clip(score, 5, 95)
    return {"name": "💹 Earnings Analyst", "score": round(score, 1), "signals": signals,
            "bias": "BULL" if score > 55 else "BEAR" if score < 45 else "NEUTRAL",
            "confidence": round(abs(score - 50) * 2, 1)}


def run_agent_flow_analyst(df: pd.DataFrame) -> dict:
    """Institutional Flow Analyst — volume & money flow."""
    score = 50.0
    signals = []

    if df.empty or len(df) < 10:
        return {"name": "🌊 Flow Analyst", "score": 50.0, "signals": [],
                "bias": "NEUTRAL", "confidence": 0}

    df2 = compute_indicators(df.copy())

    # OBV trend
    if "OBV" in df2.columns and len(df2) >= 10:
        obv_now  = df2["OBV"].iloc[-1]
        obv_prev = df2["OBV"].iloc[-10]
        obv_pct  = (obv_now - obv_prev) / (abs(obv_prev) + 1) * 100
        if obv_pct > 5:   score += 12; signals.append((f"OBV +{obv_pct:.1f}% → accumulation institutionnelle", "+12", "#10b981"))
        elif obv_pct < -5: score -= 12; signals.append((f"OBV {obv_pct:.1f}% → distribution institutionnelle", "-12", "#ef4444"))
        else: signals.append((f"OBV stable → flux neutre", "0", "#3b82f6"))

    # Volume vs SMA
    if "Vol_SMA" in df2.columns:
        last_vol = df2["Volume"].iloc[-1]
        vol_sma  = df2["Vol_SMA"].iloc[-1]
        last_close = df2["Close"].iloc[-1]
        prev_close = df2["Close"].iloc[-2] if len(df2) > 1 else last_close
        if not np.isnan(vol_sma) and vol_sma > 0:
            vol_ratio = last_vol / vol_sma
            if vol_ratio > 1.5 and last_close > prev_close:
                score += 10; signals.append((f"Volume x{vol_ratio:.1f} sur hausse → achat fort", "+10", "#10b981"))
            elif vol_ratio > 1.5 and last_close < prev_close:
                score -= 10; signals.append((f"Volume x{vol_ratio:.1f} sur baisse → vente forte", "-10", "#ef4444"))
            elif vol_ratio < 0.5:
                signals.append(("Volume très faible → conviction absente", "0", "#fbbf24"))

    # Institutional flow indicator (proxy)
    inst_flow = ECONOMIC_INDICATORS.get("Flux institutionnels", {}).get("value", 50)
    if inst_flow > 65: score += 8; signals.append((f"Flux instit.={inst_flow} → entrées", "+8", "#10b981"))
    elif inst_flow < 40: score -= 8; signals.append((f"Flux instit.={inst_flow} → sorties", "-8", "#ef4444"))

    score = np.clip(score, 5, 95)
    return {"name": "🌊 Flow Analyst", "score": round(score, 1), "signals": signals,
            "bias": "BULL" if score > 55 else "BEAR" if score < 45 else "NEUTRAL",
            "confidence": round(abs(score - 50) * 2, 1)}


def run_agent_risk_analyst(df: pd.DataFrame, market_name: str) -> dict:
    """Risk / Volatility Agent — tail risk & stress."""
    score = 50.0
    signals = []

    vix   = ECONOMIC_INDICATORS.get("VIX (Peur marché)", {}).get("value", 18)
    geopo = ECONOMIC_INDICATORS.get("Tension géopolitique", {}).get("value", 5)
    polrisk = ECONOMIC_INDICATORS.get("Élections/Risque politique", {}).get("value", 5)

    if vix < 13:   score += 10; signals.append((f"VIX={vix} → complacency → risque retournement", "+10 (attention)", "#fbbf24"))
    elif vix < 18: score += 6;  signals.append((f"VIX={vix} → faible → environnement sain", "+6", "#10b981"))
    elif vix > 25: score -= 12; signals.append((f"VIX={vix} → peur → vente panique possible", "-12", "#ef4444"))
    elif vix > 20: score -= 6;  signals.append((f"VIX={vix} → incertitude", "-6", "#fbbf24"))

    if geopo > 7:   score -= 10; signals.append((f"Géopolitique={geopo}/10 → risque élevé", "-10", "#ef4444"))
    elif geopo > 5: score -= 4;  signals.append((f"Géopolitique={geopo}/10 → risque modéré", "-4", "#fbbf24"))

    if polrisk > 6: score -= 6; signals.append((f"Risque politique={polrisk}/10 → élevé", "-6", "#ef4444"))

    # ATR-based vol
    if not df.empty and len(df) >= 14:
        df2 = compute_indicators(df.copy())
        atr = df2["ATR"].iloc[-1] if "ATR" in df2.columns else np.nan
        close = df2["Close"].iloc[-1]
        if not np.isnan(atr) and close > 0:
            atr_pct = atr / close * 100
            if atr_pct > 2.0:  score -= 8; signals.append((f"ATR={atr_pct:.2f}% → forte volatilité", "-8", "#ef4444"))
            elif atr_pct < 0.5: score += 6; signals.append((f"ATR={atr_pct:.2f}% → faible volatilité", "+6", "#10b981"))
            else: signals.append((f"ATR={atr_pct:.2f}% → volatilité normale", "0", "#3b82f6"))

    score = np.clip(score, 5, 95)
    return {"name": "⚠️ Risk Analyst", "score": round(score, 1), "signals": signals,
            "bias": "BULL" if score > 55 else "BEAR" if score < 45 else "NEUTRAL",
            "confidence": round(abs(score - 50) * 2, 1)}


def run_monte_carlo_simulation(
    df: pd.DataFrame,
    agents: list,
    timeframe_minutes: int,
    n_simulations: int = 500,
) -> dict:
    """
    Monte Carlo price path simulation.
    Combines agent scores + historical volatility to simulate price paths.
    Returns distribution of outcomes.
    """
    if df.empty or len(df) < 10:
        return {}

    close = df["Close"].dropna()
    current_price = close.iloc[-1]

    # Historical vol (annualized → per-bar)
    log_returns = np.log(close / close.shift(1)).dropna()
    hist_vol = log_returns.std()
    if np.isnan(hist_vol) or hist_vol == 0:
        hist_vol = 0.001

    # Agent consensus score → drift
    valid_agents = [a for a in agents if a.get("score") is not None]
    if not valid_agents:
        return {}

    weights = {"🏛️ Macro Strategist": 0.20, "📐 Technical Analyst": 0.35,
               "📱 Sentiment Analyst": 0.15, "💹 Earnings Analyst": 0.15,
               "🌊 Flow Analyst": 0.10, "⚠️ Risk Analyst": 0.05}

    weighted_score = 0.0
    total_w = 0.0
    for agent in valid_agents:
        w = weights.get(agent["name"], 0.1)
        weighted_score += agent["score"] * w
        total_w += w
    if total_w > 0:
        weighted_score /= total_w

    # Convert score 0-100 → drift per bar (score 50 = zero drift)
    # Scale: ±50 points = ±0.5% drift per bar
    drift = (weighted_score - 50) / 100 * 0.005

    # Time horizon in bars
    n_bars = max(1, timeframe_minutes // max(len(df), 1) * 10)
    n_bars = min(n_bars, 50)

    # Run simulations
    np.random.seed(42)
    end_prices = []
    paths_sample = []  # store 50 sample paths for chart

    for i in range(n_simulations):
        returns = np.random.normal(drift, hist_vol, n_bars)
        path = current_price * np.exp(np.cumsum(returns))
        end_prices.append(path[-1])
        if i < 50:
            paths_sample.append(path)

    end_prices = np.array(end_prices)

    bull_prob  = (end_prices > current_price).mean() * 100
    bear_prob  = 100 - bull_prob
    mean_price = end_prices.mean()
    p10 = np.percentile(end_prices, 10)
    p25 = np.percentile(end_prices, 25)
    p50 = np.percentile(end_prices, 50)
    p75 = np.percentile(end_prices, 75)
    p90 = np.percentile(end_prices, 90)

    expected_return = (mean_price / current_price - 1) * 100
    max_gain = (end_prices.max() / current_price - 1) * 100
    max_loss = (end_prices.min() / current_price - 1) * 100
    vol_outcome = end_prices.std() / current_price * 100

    return {
        "current_price":    current_price,
        "mean_price":       mean_price,
        "bull_prob":        round(bull_prob, 1),
        "bear_prob":        round(bear_prob, 1),
        "expected_return":  round(expected_return, 3),
        "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        "max_gain":         round(max_gain, 2),
        "max_loss":         round(max_loss, 2),
        "vol_outcome":      round(vol_outcome, 3),
        "n_simulations":    n_simulations,
        "n_bars":           n_bars,
        "weighted_score":   round(weighted_score, 1),
        "drift_per_bar":    round(drift * 100, 4),
        "hist_vol_per_bar": round(hist_vol * 100, 4),
        "paths_sample":     paths_sample,
        "end_prices":       end_prices,
    }


def build_scenario_matrix(agents: list, base_mc: dict) -> list:
    """Generate Bull / Base / Bear scenarios."""
    if not base_mc:
        return []
    cp = base_mc["current_price"]
    scenarios = []

    for name, score_adj, prob_adj, color, icon in [
        ("🚀 Bull Case",  +15, +20, "#10b981", "📈"),
        ("📊 Base Case",    0,   0, "#3b82f6", "➡️"),
        ("🔻 Bear Case",  -15, -20, "#ef4444", "📉"),
    ]:
        adj_score = np.clip(base_mc["weighted_score"] + score_adj, 5, 95)
        drift_adj = (adj_score - 50) / 100 * 0.005
        exp_ret = drift_adj * base_mc["n_bars"] * 100
        target = round(cp * (1 + exp_ret / 100), 4)
        prob = np.clip(base_mc["bull_prob"] + prob_adj, 5, 95) if "Bull" in name else \
               np.clip(base_mc["bear_prob"] + prob_adj, 5, 95) if "Bear" in name else \
               round(base_mc["bull_prob"], 1)
        scenarios.append({
            "name": name, "color": color, "icon": icon,
            "score": round(adj_score, 1),
            "target_price": target,
            "expected_return": round(exp_ret, 3),
            "probability": round(prob, 1),
        })
    return scenarios




# ═══════════════════════════════════════════════════════════════════
# SIMULATION CHARTS
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# SIMULATION CHARTS
# ═══════════════════════════════════════════════════════════════════

def make_agent_radar(agents: list, market_name: str) -> go.Figure:
    """Radar chart of all agent scores."""
    categories = [a["name"].split(" ", 1)[1] if " " in a["name"] else a["name"] for a in agents]
    values     = [a["score"] for a in agents]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill='toself',
        fillcolor='rgba(0,212,255,0.15)',
        line=dict(color='#00d4ff', width=2),
        name="Score agents",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50]*len(cats_closed), theta=cats_closed,
        line=dict(color='rgba(255,255,255,0.2)', dash='dash', width=1),
        showlegend=False,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9, color="#8892b0"),
                            gridcolor="#1e3a5f", linecolor="#1e3a5f"),
            angularaxis=dict(tickfont=dict(size=9, color="#ccd6f6"), gridcolor="#1e3a5f"),
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117",
        title=dict(text=f"Consensus agents — {market_name}", font=dict(color="#00d4ff", size=13)),
        height=380,
        font=dict(color="#8892b0"),
        showlegend=False,
        margin=dict(t=50, b=20, l=40, r=40),
    )
    return fig


def make_monte_carlo_chart(mc: dict, market_name: str) -> go.Figure:
    """Fan chart of Monte Carlo price paths."""
    if not mc or not mc.get("paths_sample"):
        return go.Figure()

    cp     = mc["current_price"]
    n_bars = mc["n_bars"]
    x_axis = list(range(n_bars + 1))

    fig = go.Figure()

    # Draw sample paths (faded)
    paths = mc["paths_sample"]
    for i, path in enumerate(paths[:40]):
        full_path = [cp] + list(path)
        color = "rgba(0,212,255,0.06)" if full_path[-1] >= cp else "rgba(239,68,68,0.06)"
        fig.add_trace(go.Scatter(
            x=x_axis, y=full_path,
            mode='lines', line=dict(color=color, width=1),
            showlegend=False, hoverinfo='skip',
        ))

    # Percentile bands
    for pct_lo, pct_hi, color, name in [
        ("p10", "p90", "rgba(59,130,246,0.12)", "P10-P90"),
        ("p25", "p75", "rgba(59,130,246,0.25)", "P25-P75"),
    ]:
        lo_path = [cp] + [mc[pct_lo]] * n_bars
        hi_path = [cp] + [mc[pct_hi]] * n_bars
        fig.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1],
            y=hi_path + lo_path[::-1],
            fill='toself', fillcolor=color,
            line=dict(width=0), name=name, hoverinfo='skip',
        ))

    # Median and mean paths
    median_path = [cp] + [mc["p50"]] * n_bars
    mean_path   = [cp] + [mc["mean_price"]] * n_bars
    fig.add_trace(go.Scatter(x=x_axis, y=median_path, mode='lines',
        line=dict(color="#fbbf24", width=2, dash='dash'), name="Médiane"))
    fig.add_trace(go.Scatter(x=x_axis, y=mean_path, mode='lines',
        line=dict(color="#00d4ff", width=2), name="Moyenne"))
    # Current price line
    fig.add_hline(y=cp, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text=f"Prix actuel: {cp:,.3f}", annotation_font_color="#ccd6f6")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=dict(text=f"🎲 Monte Carlo — {mc['n_simulations']} simulations — {market_name}",
                   font=dict(color="#00d4ff", size=13)),
        xaxis_title="Barres à venir", yaxis_title="Prix simulé",
        height=400,
        font=dict(color="#8892b0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig


def make_probability_distribution(mc: dict, market_name: str) -> go.Figure:
    """Histogram of simulated end prices with bull/bear split."""
    if not mc or "end_prices" not in mc:
        return go.Figure()

    ep = mc["end_prices"]
    cp = mc["current_price"]

    bull_prices = ep[ep >= cp]
    bear_prices = ep[ep <  cp]

    fig = go.Figure()
    if len(bear_prices):
        fig.add_trace(go.Histogram(x=bear_prices, nbinsx=30, name="Baisse 📉",
            marker_color="rgba(239,68,68,0.7)", showlegend=True))
    if len(bull_prices):
        fig.add_trace(go.Histogram(x=bull_prices, nbinsx=30, name="Hausse 📈",
            marker_color="rgba(16,185,129,0.7)", showlegend=True))

    fig.add_vline(x=cp,              line_dash="dash", line_color="white",
                  annotation_text="Actuel", annotation_font_color="#ccd6f6", annotation_position="top")
    fig.add_vline(x=mc["mean_price"], line_dash="dot", line_color="#00d4ff",
                  annotation_text="Moyenne", annotation_font_color="#00d4ff")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=dict(text=f"Distribution des prix finaux simulés — {market_name}",
                   font=dict(color="#00d4ff", size=13)),
        barmode="overlay",
        xaxis_title="Prix final simulé", yaxis_title="Fréquence",
        height=320,
        font=dict(color="#8892b0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig


def make_probability_heatmap(all_predictions: dict, markets: dict) -> go.Figure:
    """Heatmap: markets × timeframes colored by bull probability."""
    tf_list  = ["5 min", "15 min", "60 min", "240 min"]
    mkt_list = list(markets.keys())

    z = []
    text = []
    for mkt in mkt_list:
        row_z = []
        row_t = []
        for tf in tf_list:
            p = all_predictions.get(mkt, {}).get(tf, {})
            bp = p.get("bull_prob", 50)
            d  = p.get("direction", "—")
            row_z.append(bp)
            row_t.append(f"{d[:4]}<br>{bp}%")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=tf_list, y=[f"{markets[m]['icon']} {m}" for m in mkt_list],
        text=text, texttemplate="%{text}",
        colorscale=[
            [0.0,  "#7f1d1d"],
            [0.35, "#ef4444"],
            [0.5,  "#1e3a5f"],
            [0.65, "#10b981"],
            [1.0,  "#064e3b"],
        ],
        zmid=50, zmin=20, zmax=80,
        textfont=dict(size=10, color="white"),
        showscale=True,
        colorbar=dict(title=dict(text="Bull%", font=dict(color="#8892b0")), tickfont=dict(color="#8892b0")),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=dict(text="🌡️ Heatmap Probabilités — Marchés × Horizons", font=dict(color="#00d4ff", size=14)),
        height=320,
        font=dict(color="#8892b0"),
        xaxis=dict(side="top"),
        margin=dict(t=80, b=20, l=160, r=60),
    )
    return fig


def make_scenario_chart(scenarios: list, mc: dict) -> go.Figure:
    """Bull / Base / Bear scenario visualization — two separate subplots."""
    if not scenarios or not mc:
        return go.Figure()

    cp = mc["current_price"]
    names   = [s["name"] for s in scenarios]
    targets = [s["target_price"] for s in scenarios]
    probs   = [s["probability"] for s in scenarios]
    colors  = [s["color"] for s in scenarios]
    rets    = [s["expected_return"] for s in scenarios]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Prix cible par scénario", "Probabilité par scénario"),
        column_widths=[0.55, 0.45],
        specs=[[{"type": "xy"}, {"type": "domain"}]],
    )

    # Price targets bar
    fig.add_trace(go.Bar(
        x=names, y=targets,
        marker_color=colors,
        text=[f"{t:,.3f}<br>({r:+.3f}%)" for t,r in zip(targets,rets)],
        textposition="outside",
        textfont=dict(size=10, color="white"),
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(y=cp, line_dash="dot", line_color="rgba(255,255,255,0.4)",
                  annotation_text=f"Actuel: {cp:,.3f}",
                  annotation_font_color="#ccd6f6", row=1, col=1)

    # Probability pie
    fig.add_trace(go.Pie(
        labels=names, values=probs,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hole=0.4,
        textfont=dict(size=10),
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=dict(text="📊 Scénarios Bull / Base / Bear", font=dict(color="#00d4ff", size=13)),
        height=380,
        font=dict(color="#8892b0"),
        margin=dict(t=60, b=30, l=40, r=40),
    )
    fig.update_yaxes(gridcolor="#1e3a5f", row=1, col=1)
    return fig





# ═══════════════════════════════════════════════════════════════════
# MIROFISH SWARM INTELLIGENCE ENGINE v2
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# MIROFISH SWARM INTELLIGENCE ENGINE
# Inspired by the Chinese quant's SPX simulator (18k GitHub stars)
# 40-year historical reaction patterns · VADER sentiment · 4-agent swarm
# ═══════════════════════════════════════════════════════════════════════

# ── Reddit public RSS sources (no API key needed) ──
REDDIT_SOURCES = {
    "r/wallstreetbets":   "https://www.reddit.com/r/wallstreetbets/hot/.rss",
    "r/investing":        "https://www.reddit.com/r/investing/hot/.rss",
    "r/stocks":           "https://www.reddit.com/r/stocks/hot/.rss",
    "r/economics":        "https://www.reddit.com/r/economics/hot/.rss",
    "r/worldnews":        "https://www.reddit.com/r/worldnews/hot/.rss",
    "r/geopolitics":      "https://www.reddit.com/r/geopolitics/hot/.rss",
    "r/options":          "https://www.reddit.com/r/options/hot/.rss",
    "r/MacroEconomics":   "https://www.reddit.com/r/MacroEconomics/hot/.rss",
}

# Google News RSS (titre articles journaux financiers)
GOOGLE_NEWS_QUERIES = {
    "Nasdaq QQQ":      "https://news.google.com/rss/search?q=Nasdaq+QQQ+stock&hl=fr&gl=FR&ceid=FR:fr",
    "S&P500 SPY":      "https://news.google.com/rss/search?q=SP500+SPY+market&hl=fr&gl=FR&ceid=FR:fr",
    "Fed Reserve":     "https://news.google.com/rss/search?q=Federal+Reserve+interest+rates&hl=fr&gl=FR&ceid=FR:fr",
    "Oil Brent WTI":   "https://news.google.com/rss/search?q=oil+Brent+WTI+price&hl=fr&gl=FR&ceid=FR:fr",
    "Gold XAU":        "https://news.google.com/rss/search?q=gold+XAU+price&hl=fr&gl=FR&ceid=FR:fr",
    "Euronext STOXX":  "https://news.google.com/rss/search?q=Euronext+STOXX+CAC40&hl=fr&gl=FR&ceid=FR:fr",
    "Geopolitique":    "https://news.google.com/rss/search?q=geopolitique+guerre+sanctions&hl=fr&gl=FR&ceid=FR:fr",
    "Inflation BCE":   "https://news.google.com/rss/search?q=inflation+BCE+ECB+taux&hl=fr&gl=FR&ceid=FR:fr",
}

# MiroFish historical SPX reaction patterns (40 years distilled)
MIROFISH_HISTORICAL_REACTIONS = {
    # Event type → (avg_impact_pct, std_dev, typical_duration_bars)
    "fed_hawkish":          (-0.8, 0.6, 12),
    "fed_dovish":           (+1.2, 0.7, 15),
    "fed_hold":             (+0.3, 0.4, 8),
    "inflation_hot":        (-1.1, 0.8, 10),
    "inflation_cool":       (+1.5, 0.9, 12),
    "jobs_strong":          (-0.4, 0.5, 8),
    "jobs_weak":            (-0.9, 0.7, 10),
    "gdp_beat":             (+0.8, 0.5, 10),
    "gdp_miss":             (-0.7, 0.6, 8),
    "geopolitical_shock":   (-1.8, 1.2, 6),
    "geopolitical_ease":    (+0.9, 0.6, 8),
    "oil_spike":            (-0.6, 0.5, 8),
    "oil_crash":            (-0.3, 0.4, 6),
    "tech_earnings_beat":   (+2.1, 1.0, 15),
    "tech_earnings_miss":   (-2.8, 1.3, 12),
    "china_risk":           (-1.2, 0.9, 8),
    "market_euphoria":      (+0.5, 0.3, 20),
    "market_panic":         (-2.5, 1.5, 5),
    "dollar_rally":         (-0.6, 0.4, 10),
    "vix_spike":            (-1.5, 1.0, 6),
    "liquidity_injection":  (+1.8, 0.8, 20),
    "tariff_news":          (-0.9, 0.7, 8),
    "election_risk":        (-0.4, 0.6, 15),
}

# Keyword → event mapping (for theme detection)
THEME_KEYWORDS = {
    "fed_hawkish":       ["hike","hawkish","tighten","rate rise","higher rates","restrictive","inflation fight"],
    "fed_dovish":        ["cut","dovish","pivot","easing","lower rates","accommodation","pause"],
    "fed_hold":          ["hold","unchanged","steady","pause","wait","data dependent"],
    "inflation_hot":     ["inflation surge","CPI hot","inflation high","price spike","hyperinflation"],
    "inflation_cool":    ["inflation cool","CPI miss","deflation","disinflation","price drop"],
    "geopolitical_shock":["war","attack","invasion","conflict","sanction","missile","escalat","strike","terror"],
    "geopolitical_ease": ["ceasefire","peace","deal","truce","negotiat","diplomacy"],
    "oil_spike":         ["oil spike","crude surge","OPEC cut","energy crisis","supply shock","WTI high","Brent rise"],
    "oil_crash":         ["oil crash","crude drop","OPEC increase","oversupply","demand collapse"],
    "tech_earnings_beat":["beat","earnings surprise","profit beat","revenue beat","guidance raise","EPS beat"],
    "tech_earnings_miss":["miss","earnings miss","profit miss","guidance cut","EPS miss","warning"],
    "china_risk":        ["china","taiwan","beijing","trade war","decoupling","xi jinping"],
    "market_panic":      ["crash","sell-off","panic","circuit breaker","black swan","collapse","rout"],
    "market_euphoria":   ["rally","bull run","all-time high","ath","moon","fomo","euphoria"],
    "dollar_rally":      ["dollar surge","DXY high","USD strong","dollar index"],
    "vix_spike":         ["vix spike","fear index","volatility surge","options hedge"],
    "tariff_news":       ["tariff","trade war","import duty","protectionism","trade barrier"],
    "election_risk":     ["election","vote","poll","president","congress","senate","political risk"],
}

ASSET_KEYWORDS = {
    "NASDAQ 100":    ["nasdaq","qqq","tech","ndx","faang","semiconductor","ai stocks","growth"],
    "S&P 500":       ["spx","spy","sp500","s&p","large cap","index fund","market cap"],
    "Or (Gold)":     ["gold","xau","gld","precious metal","safe haven","bullion","gold price"],
    "CAC 40":        ["cac","cac40","paris","euronext paris","french stocks","lvmh","total"],
    "Euronext 600":  ["euronext","stoxx","stoxx600","europe stocks","dax","ftse","european"],
    "Pétrole (WTI)": ["oil","wti","crude","brent","opec","energy","petroleum","barrel"],
}


def vader_score(text: str) -> float:
    """VADER compound score. Returns float -1 to +1."""
    if not text or not VADER_AVAILABLE or _vader is None:
        # Fallback to TextBlob
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return 0.0
    try:
        return _vader.polarity_scores(str(text))["compound"]
    except Exception:
        return 0.0


def vader_label(score: float) -> str:
    """VADER rule: >+0.1 HAUSSIER, <-0.1 BAISSIER, else NEUTRE."""
    if score > 0.10:   return "HAUSSIER"
    elif score < -0.10: return "BAISSIER"
    return "NEUTRE"


@st.cache_data(ttl=600)
def fetch_reddit_sentiment(asset_name: str = "") -> list:
    """Fetch Reddit posts via public RSS, score with VADER."""
    articles = []
    asset_kw = ASSET_KEYWORDS.get(asset_name, [])

    for source, url in REDDIT_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                pub     = entry.get("published", "")
                if not title or len(title) < 5:
                    continue
                # Clean HTML
                try:
                    from bs4 import BeautifulSoup as _BS4
                    summary_clean = _BS4(summary, "html.parser").get_text()[:300]
                except Exception:
                    summary_clean = summary[:300]

                text  = title + " " + summary_clean
                score = vader_score(text)
                label = vader_label(score)

                # Relevance boost for asset-specific keywords
                text_lower = text.lower()
                relevant = any(kw in text_lower for kw in asset_kw) if asset_kw else True

                articles.append({
                    "source":    source,
                    "title":     title[:150],
                    "polarity":  score,
                    "sentiment": label.lower().replace("haussier","positif").replace("baissier","négatif"),
                    "vader_label": label,
                    "published": pub[:30],
                    "relevant":  relevant,
                    "type":      "reddit",
                })
        except Exception:
            continue
    return articles


@st.cache_data(ttl=600)
def fetch_google_news_sentiment(asset_name: str = "") -> list:
    """Fetch Google News RSS article titles, score with VADER."""
    articles = []
    asset_kw = ASSET_KEYWORDS.get(asset_name, [])

    for query_name, url in GOOGLE_NEWS_QUERIES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:6]:
                title = entry.get("title", "")
                pub   = entry.get("published", "")
                if not title:
                    continue
                score = vader_score(title)
                label = vader_label(score)
                text_lower = title.lower()
                relevant = any(kw in text_lower for kw in asset_kw) if asset_kw else True

                articles.append({
                    "source":      f"Google News: {query_name}",
                    "title":       title[:150],
                    "polarity":    score,
                    "sentiment":   label.lower().replace("haussier","positif").replace("baissier","négatif"),
                    "vader_label": label,
                    "published":   pub[:30],
                    "relevant":    relevant,
                    "type":        "google_news",
                })
        except Exception:
            continue
    return articles


def detect_themes(texts: list, top_n: int = 3) -> list:
    """
    Detect top emerging themes from a list of texts.
    Returns list of (theme, count, impact_desc, price_impact_pct).
    """
    theme_hits = Counter()
    for text in texts:
        text_lower = text.lower()
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                theme_hits[theme] += 1

    top_themes = theme_hits.most_common(top_n)
    result = []
    for theme, count in top_themes:
        reaction = MIROFISH_HISTORICAL_REACTIONS.get(theme, (0, 0.3, 8))
        avg_imp, std_imp, duration = reaction
        # Human-readable labels
        labels = {
            "fed_hawkish":          ("🏛️ Fed Hawkish",          f"Pression baissière attendue {avg_imp:.1f}% sur {duration} bars"),
            "fed_dovish":           ("🕊️ Fed Dovish/Pivot",      f"Rebond attendu +{avg_imp:.1f}% sur {duration} bars"),
            "fed_hold":             ("⏸️ Fed Pause",             f"Réaction modérée {avg_imp:+.1f}% sur {duration} bars"),
            "inflation_hot":        ("🔥 Inflation Élevée",      f"Correction probable {avg_imp:.1f}% sur {duration} bars"),
            "inflation_cool":       ("❄️ Inflation Refroidit",   f"Rally attendu +{avg_imp:.1f}% sur {duration} bars"),
            "geopolitical_shock":   ("💥 Choc Géopolitique",     f"Sell-off {avg_imp:.1f}% court terme sur {duration} bars"),
            "geopolitical_ease":    ("🕊️ Détente Géopolitique",  f"Rebond +{avg_imp:.1f}% sur {duration} bars"),
            "oil_spike":            ("🛢️ Oil Spike",             f"Impact marché {avg_imp:.1f}% sur {duration} bars"),
            "oil_crash":            ("📉 Oil Crash",             f"Pression {avg_imp:.1f}% sur énergie sur {duration} bars"),
            "tech_earnings_beat":   ("🚀 Earnings Beat Tech",    f"Hausse +{avg_imp:.1f}% sur {duration} bars"),
            "tech_earnings_miss":   ("⚠️ Earnings Miss Tech",    f"Correction {avg_imp:.1f}% sur {duration} bars"),
            "china_risk":           ("🇨🇳 Risque Chine",          f"Pression {avg_imp:.1f}% marchés mondiaux"),
            "market_panic":         ("😱 Panique Marché",        f"Sell-off violent {avg_imp:.1f}% sur {duration} bars"),
            "market_euphoria":      ("🎉 Euphorie Marché",       f"Momentum +{avg_imp:.1f}% mais retournement risqué"),
            "dollar_rally":         ("💵 Dollar Fort",           f"Pression {avg_imp:.1f}% sur commodités"),
            "vix_spike":            ("⚡ VIX Spike",             f"Volatilité extrême {avg_imp:.1f}% sur {duration} bars"),
            "tariff_news":          ("🚧 Droits de Douane",      f"Pression {avg_imp:.1f}% sur exports/imports"),
            "election_risk":        ("🗳️ Risque Politique",      f"Incertitude {avg_imp:.1f}% sur {duration} bars"),
            "jobs_strong":          ("💼 Emploi Fort",           f"Signal mixte {avg_imp:.1f}% sur {duration} bars"),
            "liquidity_injection":  ("💧 Injection Liquidité",   f"Soutien +{avg_imp:.1f}% sur {duration} bars"),
        }
        label_icon, impact_desc = labels.get(theme, (f"📌 {theme}", f"{avg_imp:+.1f}% attendu"))
        result.append({
            "theme":        theme,
            "label":        label_icon,
            "count":        count,
            "impact_desc":  impact_desc,
            "avg_impact":   avg_imp,
            "std_impact":   std_imp,
            "duration":     duration,
        })
    return result


def mirofish_swarm_simulation(
    df: pd.DataFrame,
    all_texts: list,
    fund_score: dict,
    tw_score: dict,
    asset_name: str,
    timeframe_minutes: int,
    n_simulations: int = 1000,
) -> dict:
    """
    Full MiroFish Swarm Simulation:
    - VADER compound score on all texts
    - 4 agents voting in swarm
    - Monte Carlo 1000 paths with sentiment-biased drift
    - Probability table for price moves
    """
    # ── VADER aggregate score ──
    vader_scores = [vader_score(t) for t in all_texts if t and len(t) > 3]
    if vader_scores:
        vader_mean     = float(np.mean(vader_scores))
        vader_median   = float(np.median(vader_scores))
        vader_std      = float(np.std(vader_scores))
        vader_positive = sum(1 for s in vader_scores if s > 0.10)
        vader_negative = sum(1 for s in vader_scores if s < -0.10)
        vader_neutral  = len(vader_scores) - vader_positive - vader_negative
    else:
        vader_mean = vader_median = vader_std = 0.0
        vader_positive = vader_negative = vader_neutral = 0

    vader_verdict = vader_label(vader_mean)

    # ── Theme detection ──
    themes = detect_themes(all_texts, top_n=3)

    # ── Theme-adjusted drift ──
    theme_drift = 0.0
    for th in themes:
        # Weight by count and historical impact
        theme_drift += th["avg_impact"] * 0.001 * min(th["count"], 5)

    # ── 4 Swarm Agents ──
    agents_votes = {}

    # Agent 1: Sentiment Analyst (VADER-based)
    sent_score = 50 + vader_mean * 200
    sent_score = float(np.clip(sent_score, 5, 95))
    agents_votes["📱 Sentiment (VADER)"] = {
        "score": round(sent_score, 1),
        "basis": f"VADER compound={vader_mean:+.4f} | {vader_positive}📈 {vader_negative}📉 {vader_neutral}➡️",
        "weight": 0.30,
    }

    # Agent 2: TA Quant (RSI + MACD simulation)
    ta_score = 50.0
    ta_basis = "TA"
    if not df.empty and len(df) >= 20:
        df2  = compute_indicators(df.copy())
        last = df2.iloc[-1]
        rsi  = last.get("RSI", 50)
        macd = last.get("MACD", 0)
        msig = last.get("MACD_signal", 0)
        if not np.isnan(rsi):
            if rsi < 30:   ta_score += 18
            elif rsi > 70: ta_score -= 18
            elif rsi > 55: ta_score += 6
            elif rsi < 45: ta_score -= 6
        if not np.isnan(macd):
            if macd > msig: ta_score += 10
            else:           ta_score -= 10
        ema9  = last.get("EMA9", last["Close"])
        close = last["Close"]
        if not np.isnan(ema9):
            if close > ema9: ta_score += 8
            else:            ta_score -= 8
        ta_score = float(np.clip(ta_score, 5, 95))
        ta_basis = f"RSI={rsi:.0f} | MACD={'↑' if macd>msig else '↓'} | Prix {'>' if close>ema9 else '<'} EMA9"
    agents_votes["📐 TA Quant"] = {
        "score": round(ta_score, 1),
        "basis": ta_basis,
        "weight": 0.30,
    }

    # Agent 3: Macro Strategist (economic context)
    macro_score = fund_score.get("eco_score", 50)
    agents_votes["🏛️ Macro Strategist"] = {
        "score": round(macro_score, 1),
        "basis": f"Eco score={macro_score:.0f} | VIX={ECONOMIC_INDICATORS.get('VIX (Peur marché)',{}).get('value',18)} | Taux Fed={ECONOMIC_INDICATORS.get('Taux Fed (FFR)',{}).get('value',5.25)}%",
        "weight": 0.20,
    }

    # Agent 4: Earnings/Events Analyst (MiroFish historical reactions)
    events_score = 50.0
    events_basis = "Aucun événement majeur détecté"
    if themes:
        top_theme = themes[0]
        reaction  = top_theme["avg_impact"]
        if reaction > 0:   events_score = 50 + min(reaction * 8, 35)
        elif reaction < 0: events_score = 50 + max(reaction * 8, -35)
        events_basis = f"{top_theme['label']} ({top_theme['count']} mentions) → {top_theme['impact_desc']}"
    agents_votes["💹 Earnings/Events (MiroFish)"] = {
        "score": round(float(np.clip(events_score, 5, 95)), 1),
        "basis": events_basis,
        "weight": 0.20,
    }

    # ── Swarm consensus ──
    total_w   = sum(a["weight"] for a in agents_votes.values())
    consensus = sum(a["score"] * a["weight"] for a in agents_votes.values()) / total_w
    consensus = float(np.clip(consensus, 5, 95))

    # ── Monte Carlo 1000 simulations ──
    current_price = df["Close"].iloc[-1] if not df.empty else 100.0

    # Historical volatility
    if not df.empty and len(df) >= 10:
        log_ret  = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        hist_vol = float(log_ret.std())
        if np.isnan(hist_vol) or hist_vol == 0:
            hist_vol = 0.0015
    else:
        hist_vol = 0.0015

    # Drift = consensus bias + theme drift
    base_drift   = (consensus - 50) / 100 * 0.006
    total_drift  = base_drift + theme_drift * 0.001
    # n_bars for timeframe
    n_bars = max(1, min(timeframe_minutes // 5, 60))

    np.random.seed(42)
    end_prices  = []
    paths_10    = []  # 10 sample paths for chart

    for i in range(n_simulations):
        shocks   = np.random.normal(total_drift, hist_vol, n_bars)
        path_end = current_price * np.exp(np.sum(shocks))
        end_prices.append(path_end)
        if i < 10:
            path = current_price * np.exp(np.cumsum(shocks))
            paths_10.append(list(path))

    end_prices = np.array(end_prices)

    # ── Probability table ──
    def prob_above(pct): return float((end_prices > current_price * (1 + pct/100)).mean() * 100)
    def prob_below(pct): return float((end_prices < current_price * (1 - pct/100)).mean() * 100)

    prob_table = {
        "+0.5%":  round(prob_above(0.5), 1),
        "+1.0%":  round(prob_above(1.0), 1),
        "+2.0%":  round(prob_above(2.0), 1),
        "-0.5%":  round(prob_below(0.5), 1),
        "-1.0%":  round(prob_below(1.0), 1),
        "-2.0%":  round(prob_below(2.0), 1),
    }

    bull_prob = round(float((end_prices > current_price).mean() * 100), 1)
    bear_prob = round(100 - bull_prob, 1)

    # ── Trade recommendation ──
    if consensus >= 62 and vader_mean > 0.05:
        trade_signal = "LONG 📈"
        trade_color  = "#10b981"
        trade_conf   = round(min((consensus - 50) * 2 + abs(vader_mean) * 50, 92), 1)
        target_price = round(current_price * (1 + abs(total_drift) * n_bars * 3), 4)
        sl_price     = round(current_price * (1 - hist_vol * 3), 4)
    elif consensus <= 38 and vader_mean < -0.05:
        trade_signal = "SHORT 📉"
        trade_color  = "#ef4444"
        trade_conf   = round(min((50 - consensus) * 2 + abs(vader_mean) * 50, 92), 1)
        target_price = round(current_price * (1 - abs(total_drift) * n_bars * 3), 4)
        sl_price     = round(current_price * (1 + hist_vol * 3), 4)
    else:
        trade_signal = "WAIT ⏳"
        trade_color  = "#fbbf24"
        trade_conf   = round(50 - abs(consensus - 50), 1)
        target_price = round(current_price, 4)
        sl_price     = round(current_price, 4)

    exp_return = round((end_prices.mean() / current_price - 1) * 100, 3)
    p10 = float(np.percentile(end_prices, 10))
    p50 = float(np.percentile(end_prices, 50))
    p90 = float(np.percentile(end_prices, 90))

    return {
        # VADER
        "vader_mean":     round(vader_mean, 4),
        "vader_median":   round(vader_median, 4),
        "vader_std":      round(vader_std, 4),
        "vader_verdict":  vader_verdict,
        "vader_positive": vader_positive,
        "vader_negative": vader_negative,
        "vader_neutral":  vader_neutral,
        "n_texts":        len(vader_scores),
        # Themes
        "themes":         themes,
        # Agents
        "agents":         agents_votes,
        "consensus":      round(consensus, 1),
        # MC
        "current_price":  current_price,
        "bull_prob":      bull_prob,
        "bear_prob":      bear_prob,
        "exp_return":     exp_return,
        "prob_table":     prob_table,
        "paths_10":       paths_10,
        "n_bars":         n_bars,
        "p10": round(p10, 4), "p50": round(p50, 4), "p90": round(p90, 4),
        "hist_vol":       round(hist_vol * 100, 4),
        "drift":          round(total_drift * 100, 5),
        # Trade
        "trade_signal":   trade_signal,
        "trade_color":    trade_color,
        "trade_conf":     trade_conf,
        "target_price":   target_price,
        "sl_price":       sl_price,
    }


def make_mirofish_mc_chart(result: dict, asset_name: str, timeframe: str) -> go.Figure:
    """10 MC paths fan chart + price levels."""
    fig = go.Figure()
    cp     = result["current_price"]
    n_bars = result["n_bars"]
    x_ax   = list(range(n_bars + 1))

    for i, path in enumerate(result["paths_10"]):
        full = [cp] + path
        color = "rgba(16,185,129,0.25)" if full[-1] >= cp else "rgba(239,68,68,0.25)"
        fig.add_trace(go.Scatter(x=x_ax, y=full, mode="lines",
            line=dict(color=color, width=1.5), showlegend=False, hoverinfo="skip"))

    # Median path
    fig.add_trace(go.Scatter(
        x=x_ax, y=[cp] + [result["p50"]] * n_bars, mode="lines",
        line=dict(color="#fbbf24", width=2.5, dash="dash"), name="Médiane P50"))
    # P10/P90 band
    fig.add_trace(go.Scatter(
        x=x_ax + x_ax[::-1],
        y=[cp] + [result["p90"]] * n_bars + ([cp] + [result["p10"]] * n_bars)[::-1],
        fill="toself", fillcolor="rgba(59,130,246,0.10)",
        line=dict(width=0), name="P10-P90", hoverinfo="skip"))

    fig.add_hline(y=cp, line_dash="dot", line_color="rgba(255,255,255,0.4)",
                  annotation_text=f"Actuel {cp:,.3f}", annotation_font_color="#ccd6f6")
    if result["target_price"] != cp:
        tgt_color = "#10b981" if result["trade_signal"].startswith("L") else "#ef4444"
        fig.add_hline(y=result["target_price"], line_dash="dot", line_color=tgt_color,
                      annotation_text=f"Cible {result['target_price']:,.3f}",
                      annotation_font_color=tgt_color)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=dict(text=f"🎲 MiroFish MC — {asset_name} | {timeframe} | 1000 sims",
                   font=dict(color="#00d4ff", size=13)),
        height=360, font=dict(color="#8892b0"),
        xaxis_title="Barres", yaxis_title="Prix simulé",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=30, l=50, r=20),
    )
    return fig


def generate_mirofish_pdf(
    result: dict,
    asset_name: str,
    timeframe: str,
    custom_tweets: str = "",
) -> bytes:
    """
    Generate 1-page professional MiroFish report PDF.
    Style: trader pro FR, emojis in text replaced with ascii.
    """
    now = datetime.now(pytz.timezone("Europe/Paris"))
    now_str = now.strftime("%d/%m/%Y %H:%M:%S")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # ── Header banner ──
    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _p("MIROFISH SWARM INTELLIGENCE REPORT"), ln=True, fill=True, align="C")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_fill_color(25, 75, 150)
    pdf.cell(0, 5, _p(
        f"Actif: {asset_name}  |  Timeframe: {timeframe}  |  "
        f"Genere le {now_str} (Paris)  |  "
        f"1000 simulations Monte Carlo  |  VADER NLP"
    ), ln=True, fill=True, align="C")
    pdf.ln(3)

    # ── Section 1: VADER Sentiment ──
    verdict = result["vader_verdict"]
    vcolor = (0,160,80) if verdict=="HAUSSIER" else (200,40,40) if verdict=="BAISSIER" else (40,80,200)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 245)
    pdf.set_text_color(20, 50, 130)
    pdf.cell(0, 6, _p("1. SENTIMENT VADER — SCORE MOYEN"), ln=True, fill=True)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*vcolor)
    vader_icon = ">>" if verdict=="HAUSSIER" else "<<" if verdict=="BAISSIER" else "--"
    pdf.cell(0, 8, _p(
        f"  {vader_icon} {verdict}   Compound: {result['vader_mean']:+.4f}   "
        f"(seuil: >+0.10=HAUSSIER, <-0.10=BAISSIER)"
    ), ln=True)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 80)
    pdf.cell(0, 5, _p(
        f"  Textes analyses: {result['n_texts']}  |  "
        f"Positifs: {result['vader_positive']}  |  "
        f"Negatifs: {result['vader_negative']}  |  "
        f"Neutres: {result['vader_neutral']}  |  "
        f"Std dev: {result['vader_std']:.4f}"
    ), ln=True)
    pdf.ln(2)

    # ── Section 2: Top 3 Themes ──
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 245)
    pdf.set_text_color(20, 50, 130)
    pdf.cell(0, 6, _p("2. TOP 3 THEMES EMERGENTS & IMPACT PRIX"), ln=True, fill=True)

    themes = result.get("themes", [])
    if themes:
        for i, th in enumerate(themes[:3], 1):
            imp   = th["avg_impact"]
            color = (0,140,70) if imp > 0 else (180,30,30)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*color)
            pdf.cell(8, 5, _p(f"#{i}"), border=0)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5, _p(
                f"{th['label']}  ({th['count']} mentions)  -->  {th['impact_desc']}"
            ), ln=True)
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 5, _p("  Aucun theme dominant detecte (donnees insuffisantes)"), ln=True)
    pdf.ln(2)

    # ── Section 3: Swarm Agents ──
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 245)
    pdf.set_text_color(20, 50, 130)
    pdf.cell(0, 6, _p("3. MIROFISH SWARM — 4 AGENTS"), ln=True, fill=True)

    pdf.set_fill_color(200, 215, 240)
    pdf.set_text_color(20, 60, 130)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(55, 5, _p("Agent"), fill=True, border=1)
    pdf.cell(20, 5, _p("Score"), fill=True, border=1, align="C")
    pdf.cell(18, 5, _p("Poids"), fill=True, border=1, align="C")
    pdf.cell(0,  5, _p("Analyse"), fill=True, border=1)
    pdf.ln()

    for agent_name, agent_data in result.get("agents", {}).items():
        sc = agent_data["score"]
        w  = agent_data["weight"]
        if sc >= 60:   pdf.set_text_color(0,140,70);  pdf.set_fill_color(245,255,248)
        elif sc <= 40: pdf.set_text_color(180,30,30);  pdf.set_fill_color(255,245,245)
        else:          pdf.set_text_color(40,80,170);  pdf.set_fill_color(245,248,255)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(55, 4, _p(agent_name[:30]), fill=True, border=1)
        pdf.cell(20, 4, _p(f"{sc:.1f}/100"), fill=True, border=1, align="C")
        pdf.cell(18, 4, _p(f"{w*100:.0f}%"), fill=True, border=1, align="C")
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(50,50,70)
        pdf.cell(0, 4, _p(agent_data["basis"][:65]), fill=True, border=1)
        pdf.ln()

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 50, 130)
    pdf.set_fill_color(200, 215, 240)
    pdf.cell(0, 6, _p(f"  CONSENSUS SWARM: {result['consensus']:.1f}/100"), ln=True, fill=True)
    pdf.ln(2)

    # ── Section 4: Monte Carlo Probabilities ──
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 245)
    pdf.set_text_color(20, 50, 130)
    pdf.cell(0, 6, _p("4. MIROFISH MONTE CARLO — 1000 SIMULATIONS"), ln=True, fill=True)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60,60,80)
    pdf.cell(0, 5, _p(
        f"  Prix actuel: {result['current_price']:,.4f}  |  "
        f"P10: {result['p10']:,.4f}  |  P50: {result['p50']:,.4f}  |  P90: {result['p90']:,.4f}  |  "
        f"Retour exp: {result['exp_return']:+.3f}%  |  Vol historique: {result['hist_vol']:.4f}%/bar"
    ), ln=True)

    # Probability table
    pt = result["prob_table"]
    pdf.set_fill_color(200, 215, 240)
    pdf.set_text_color(20, 60, 130)
    pdf.set_font("Helvetica", "B", 9)
    cols_prob = [
        ("+0.5%", pt["+0.5%"], "#10b981"),
        ("+1.0%", pt["+1.0%"], "#10b981"),
        ("+2.0%", pt["+2.0%"], "#10b981"),
        ("-0.5%", pt["-0.5%"], "#ef4444"),
        ("-1.0%", pt["-1.0%"], "#ef4444"),
        ("-2.0%", pt["-2.0%"], "#ef4444"),
    ]
    for label, _, _ in cols_prob:
        pdf.cell(32, 5, _p(label), fill=True, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "B", 10)
    for label, val, color in cols_prob:
        if color == "#10b981": pdf.set_text_color(0,140,70);  pdf.set_fill_color(240,255,245)
        else:                   pdf.set_text_color(180,30,30); pdf.set_fill_color(255,242,242)
        pdf.cell(32, 6, _p(f"{val}%"), fill=True, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60,60,80)
    pdf.cell(0, 5, _p(
        f"  HAUSSE probable: {result['bull_prob']}%   |   BAISSE probable: {result['bear_prob']}%   |   "
        f"Drift/bar: {result['drift']:+.5f}%   |   Barres simulees: {result['n_bars']}"
    ), ln=True)
    pdf.ln(2)

    # ── Section 5: Trade Recommendation ──
    sig   = result["trade_signal"]
    conf  = result["trade_conf"]
    tgt   = result["target_price"]
    sl    = result["sl_price"]
    if "LONG" in sig:   fill_rgb=(230,255,240); text_rgb=(0,140,70)
    elif "SHORT" in sig: fill_rgb=(255,235,235); text_rgb=(180,30,30)
    else:               fill_rgb=(255,250,220); text_rgb=(150,100,0)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 245)
    pdf.set_text_color(20, 50, 130)
    pdf.cell(0, 6, _p("5. VERDICT & TRADE SCALPING"), ln=True, fill=True)

    pdf.set_fill_color(*fill_rgb)
    pdf.set_text_color(*text_rgb)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, _p(
        f"  {sig}   Confiance: {conf}%   |   "
        f"Cible: {tgt:,.4f}   |   SL: {sl:,.4f}"
    ), ln=True, fill=True)

    # 4-line comment
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(60, 60, 80)
    verdict_comments = {
        "HAUSSIER": [
            f"Swarm consensus haussier {result['consensus']:.0f}/100 avec VADER {result['vader_mean']:+.4f}.",
            f"Top theme: {themes[0]['label'] if themes else 'N/A'} - pression prix favorable.",
            f"MC 1000 sims: {result['bull_prob']}% prob hausse. P90={result['p90']:,.3f} vs actuel {result['current_price']:,.3f}.",
            f"LONG recommande avec stop a {sl:,.4f}. Cible {tgt:,.4f} sur {timeframe}.",
        ],
        "BAISSIER": [
            f"Swarm consensus baissier {result['consensus']:.0f}/100 avec VADER {result['vader_mean']:+.4f}.",
            f"Top theme: {themes[0]['label'] if themes else 'N/A'} - pression vendeuse dominante.",
            f"MC 1000 sims: {result['bear_prob']}% prob baisse. P10={result['p10']:,.3f} vs actuel {result['current_price']:,.3f}.",
            f"SHORT recommande avec stop a {sl:,.4f}. Cible {tgt:,.4f} sur {timeframe}.",
        ],
        "NEUTRE": [
            f"Signal swarm mixte {result['consensus']:.0f}/100. VADER neutre {result['vader_mean']:+.4f}.",
            f"Pas de theme dominant fort - marche en attente de catalyseur.",
            f"MC 1000 sims equilibre: {result['bull_prob']}% hausse / {result['bear_prob']}% baisse.",
            f"WAIT - Eviter le trade. Attendre signal directonnel clair (RSI ou breakout niveau cle).",
        ],
    }
    comments = verdict_comments.get(verdict, verdict_comments["NEUTRE"])
    for line in comments:
        pdf.cell(0, 4, _p(f"  > {line}"), ln=True)

    pdf.ln(3)

    # ── Footer ──
    pdf.set_draw_color(100, 130, 180)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(1)
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 3, _p(
        f"MiroFish Swarm Intelligence | Sources: Twitter/X · Reddit · Google News · Yahoo Finance | "
        f"VADER NLP | Monte Carlo 1000 sims | "
        f"AVERTISSEMENT: Rapport educatif uniquement. Pas de conseil financier. | {now_str}"
    ), ln=True, align="C")

    return bytes(pdf.output())



# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>📈 Market Sentiment Predictor</h1>
        <p>Analyse multi-facteurs • Prédictions probabilistes • NASDAQ • S&P500 • Or • CAC40 • Euronext600 • Pétrole</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown('<div class="section-title">⚙️ Paramètres</div>', unsafe_allow_html=True)

        selected_market = st.selectbox(
            "📊 Marché principal",
            options=list(MARKETS.keys()),
        )

        selected_interval = st.selectbox(
            "⏱️ Intervalle graphique",
            options=list(TIMEFRAMES.keys()),
        )

        st.markdown("---")
        st.markdown('<div class="section-title">🔄 Actualisation</div>', unsafe_allow_html=True)

        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)

        if st.button("🔄 Rafraîchir maintenant", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        last_update = datetime.now(pytz.timezone("Europe/Paris")).strftime("%H:%M:%S")
        st.markdown(f'<div class="update-time">⏰ Mis à jour: {last_update}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📰 Sources actives</div>', unsafe_allow_html=True)
        for src in RSS_FEEDS:
            st.markdown(f'<span class="source-badge">{src}</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📡 Données</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-badge">Yahoo Finance</span> <span class="source-badge">yfinance API</span>', unsafe_allow_html=True)
        st.markdown('<span class="source-badge">RSS Feeds</span> <span class="source-badge">TextBlob NLP</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.warning("⚠️ Application éducative uniquement. Pas de conseil financier.")

    if auto_refresh:
        time.sleep(300)
        st.rerun()

    # ── Load data ──
    with st.spinner("🔄 Chargement des données en cours..."):
        news_articles    = fetch_news_sentiment()
        twitter_articles = fetch_twitter_sentiment(max_per_account=4)
        tw_score_data    = compute_twitter_score(twitter_articles)
        fund_data        = compute_fundamental_score(news_articles, tw_score_data)
        # MiroFish sources (cached)
        reddit_articles  = fetch_reddit_sentiment(selected_market)
        gnews_articles   = fetch_google_news_sentiment(selected_market)

        all_market_prices = {}
        all_predictions = {}

        for mkt_name, mkt_info in MARKETS.items():
            ticker  = mkt_info["ticker"]
            tf_info = TIMEFRAMES[selected_interval]
            df      = fetch_ohlc(ticker, tf_info["yf_period"], tf_info["yf_interval"])
            price_info = get_current_price(ticker)
            all_market_prices[mkt_name] = {"df": df, "price_info": price_info}

            tech = compute_technical_signal(df)
            preds = {}
            for tf_name, tf_data in TIMEFRAMES.items():
                preds[tf_name] = predict_direction(tech["score"], fund_data["combined"], tf_data["minutes"])
            all_predictions[mkt_name] = preds

    # ── Main tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Dashboard",
        "📈 Graphiques OHLC",
        "🌍 Macro & Sentiment",
        "📰 Actualités",
        "🐦 Twitter/X Intel",
        "🎯 Conseils Scalping",
        "🤖 Simulation IA",
        "🧪 MiroFish Lab",
        "📋 Sources",
    ])

    # ══════════════════════════════════════════════
    # TAB 1: DASHBOARD
    # ══════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">🏦 Vue d\'ensemble des marchés</div>', unsafe_allow_html=True)

        cols = st.columns(len(MARKETS))
        for i, (mkt_name, mkt_info) in enumerate(MARKETS.items()):
            with cols[i]:
                price_info = all_market_prices[mkt_name]["price_info"]
                preds      = all_predictions[mkt_name]
                pred_5m    = preds.get("5 min", {})
                direction  = pred_5m.get("direction", "N/A")
                score      = pred_5m.get("score", 50)

                dir_icon  = "▲" if direction == "HAUSSIER" else "▼" if direction == "BAISSIER" else "▶"
                price     = price_info.get("price", 0)
                chg_pct   = price_info.get("change_pct", 0)
                chg_class = "metric-delta-up" if chg_pct >= 0 else "metric-delta-down"
                chg_sign  = "+" if chg_pct >= 0 else ""
                sig_class = "signal-bull" if direction == "HAUSSIER" else "signal-bear" if direction == "BAISSIER" else "signal-neutral"

                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5em">{mkt_info['icon']}</div>
                    <div class="metric-label">{mkt_name}</div>
                    <div class="metric-value">{f"{price:,.2f}" if price else "—"}</div>
                    <div class="{chg_class}">{chg_sign}{chg_pct:.2f}%</div>
                    <div style="margin-top:8px">
                        <span class="{sig_class}">{dir_icon} {direction}</span>
                    </div>
                    <div style="color:#8892b0; font-size:0.75em; margin-top:5px">Score: {score:.0f}/100</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">🎯 Sentiments globaux</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        mkt_name = selected_market
        tech_sig = compute_technical_signal(all_market_prices[mkt_name]["df"])

        with col1:
            st.plotly_chart(make_sentiment_gauge(tech_sig["score"], f"Technique\n{mkt_name}"),
                            use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(make_sentiment_gauge(fund_data["eco_score"], "Score Macro-Eco"),
                            use_container_width=True, config={"displayModeBar": False})
        with col3:
            st.plotly_chart(make_sentiment_gauge(fund_data["news_score"], "Sentiment News"),
                            use_container_width=True, config={"displayModeBar": False})
        with col4:
            combined_score = fund_data["combined"] * 0.5 + tech_sig["score"] * 0.5
            st.plotly_chart(make_sentiment_gauge(combined_score, "Score Global"),
                            use_container_width=True, config={"displayModeBar": False})

        # Twitter gauge row
        col_tw1, col_tw2, col_tw3, col_tw4 = st.columns(4)
        with col_tw1:
            st.plotly_chart(make_sentiment_gauge(tw_score_data.get("score", 50), "Twitter/X Score"),
                            use_container_width=True, config={"displayModeBar": False})
        with col_tw2:
            tw_bull_pct = tw_score_data.get("bull",0) / max(tw_score_data.get("total",1),1) * 100
            st.plotly_chart(make_sentiment_gauge(tw_bull_pct, "Twitter Positif%"),
                            use_container_width=True, config={"displayModeBar": False})
        with col_tw3:
            st.plotly_chart(make_sentiment_gauge(fund_data.get("twitter_score",50), "Twitter→Combined"),
                            use_container_width=True, config={"displayModeBar": False})
        with col_tw4:
            fin_combined = fund_data["combined"] * 0.4 + tech_sig["score"] * 0.4 + tw_score_data.get("score",50) * 0.2
            st.plotly_chart(make_sentiment_gauge(fin_combined, "Score Final"),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">🔮 Prédictions par marché & horizon</div>', unsafe_allow_html=True)
        rows = []
        for mkt_name2, preds2 in all_predictions.items():
            row = {"Marché": f"{MARKETS[mkt_name2]['icon']} {mkt_name2}"}
            for tf, pred in preds2.items():
                d  = pred["direction"]
                b  = pred["bull_prob"]
                be = pred["bear_prob"]
                emoji = "📈" if d == "HAUSSIER" else "📉" if d == "BAISSIER" else "➡️"
                row[tf] = f"{emoji} {d} ({b}%↑ / {be}%↓)"
            rows.append(row)
        df_preds = pd.DataFrame(rows).set_index("Marché")
        st.dataframe(df_preds, use_container_width=True)

        st.markdown(f'<div class="section-title">📊 Probabilités — {selected_market}</div>', unsafe_allow_html=True)
        st.plotly_chart(make_probability_chart(all_predictions[selected_market]), use_container_width=True)

        st.markdown(f'<div class="section-title">⚡ Signaux techniques — {selected_market}</div>', unsafe_allow_html=True)
        signals = tech_sig.get("signals", [])
        if signals:
            sig_cols = st.columns(2)
            for i, sig in enumerate(signals):
                with sig_cols[i % 2]:
                    bull_bear = "🟢" if "BULL" in sig["signal"] else "🔴" if "BEAR" in sig["signal"] else "⚪"
                    item_class = "news-positive" if "BULL" in sig["signal"] else "news-negative" if "BEAR" in sig["signal"] else "news-neutral"
                    st.markdown(f"""
                    <div class="news-item {item_class}">
                        <div class="news-title">{bull_bear} <b>{sig['name']}</b> — {sig['value']}</div>
                        <div class="news-meta">{sig['signal']} | Poids: {sig['weight']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # TAB 2: OHLC CHARTS
    # ══════════════════════════════════════════════
    with tab2:
        st.markdown(f'<div class="section-title">📈 Graphique OHLC — {selected_market} ({selected_interval})</div>', unsafe_allow_html=True)
        df_main = all_market_prices[selected_market]["df"]
        if not df_main.empty:
            fig_main = make_candlestick_chart(df_main, selected_market, MARKETS[selected_market]["color"])
            st.plotly_chart(fig_main, use_container_width=True)
        else:
            st.error(f"Données indisponibles pour {selected_market}")

        st.markdown('<div class="section-title">📊 Tous les marchés</div>', unsafe_allow_html=True)
        cols_charts = st.columns(2)
        ci = 0
        for mkt_name3, mkt_info3 in MARKETS.items():
            if mkt_name3 == selected_market:
                continue
            df_mkt = all_market_prices[mkt_name3]["df"]
            with cols_charts[ci % 2]:
                if not df_mkt.empty:
                    fig_mini = go.Figure(go.Candlestick(
                        x=df_mkt.index[-50:],
                        open=df_mkt["Open"].iloc[-50:], high=df_mkt["High"].iloc[-50:],
                        low=df_mkt["Low"].iloc[-50:],   close=df_mkt["Close"].iloc[-50:],
                        increasing_fillcolor="#10b981", increasing_line_color="#10b981",
                        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
                    ))
                    fig_mini.update_layout(
                        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        title=f"{mkt_info3['icon']} {mkt_name3}", height=280,
                        xaxis_rangeslider_visible=False, showlegend=False,
                        margin=dict(t=40, b=20, l=20, r=20), font=dict(color="#8892b0"),
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
            ci += 1

    # ══════════════════════════════════════════════
    # TAB 3: MACRO & SENTIMENT
    # ══════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">🌍 Indicateurs économiques & géopolitiques</div>', unsafe_allow_html=True)
        st.plotly_chart(make_eco_indicators_chart(), use_container_width=True)

        st.markdown('<div class="section-title">📊 Détail des 20 indicateurs</div>', unsafe_allow_html=True)
        eco_rows = []
        for name, data in ECONOMIC_INDICATORS.items():
            eco_rows.append({
                "Indicateur": name,
                "Valeur": str(data["value"]),
                "Tendance": data["trend"],
                "Impact": data["impact"].upper(),
                "Poids": f"{data['weight']*100:.0f}%",
            })
        df_eco = pd.DataFrame(eco_rows)

        def color_impact(val):
            if val == "POSITIF":   return "color: #10b981"
            elif val == "NEGATIF" or val == "NÉGATIF": return "color: #ef4444"
            return "color: #3b82f6"

        st.dataframe(df_eco.style.applymap(color_impact, subset=["Impact"]), use_container_width=True)

        st.markdown('<div class="section-title">📊 Répartition du sentiment</div>', unsafe_allow_html=True)
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            fig_pie = go.Figure(go.Pie(
                labels=["Positif", "Negatif", "Neutre"],
                values=[fund_data["news_bull"], fund_data["news_bear"], fund_data["news_neutral"]],
                hole=0.5, marker=dict(colors=["#10b981", "#ef4444", "#3b82f6"]),
            ))
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                title="Sentiment actualites", font=dict(color="#8892b0"), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_pie2:
            pos_c = sum(1 for d in ECONOMIC_INDICATORS.values() if d["impact"] == "positif")
            neg_c = sum(1 for d in ECONOMIC_INDICATORS.values() if d["impact"] in ("négatif","negatif"))
            neu_c = sum(1 for d in ECONOMIC_INDICATORS.values() if d["impact"] == "neutre")
            fig_pie2 = go.Figure(go.Pie(
                labels=["Positif", "Negatif", "Neutre"],
                values=[pos_c, neg_c, neu_c],
                hole=0.5, marker=dict(colors=["#10b981", "#ef4444", "#3b82f6"]),
            ))
            fig_pie2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                title="Impact indicateurs macro", font=dict(color="#8892b0"), height=300)
            st.plotly_chart(fig_pie2, use_container_width=True)

    # ══════════════════════════════════════════════
    # TAB 4: NEWS
    # ══════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">📰 Actualités financières & géopolitiques</div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            filter_sentiment = st.selectbox("Filtrer:", ["Tous", "positif", "négatif", "neutre"])
        with col_f2:
            search_query = st.text_input("🔍 Rechercher:", placeholder="inflation, Fed, geopolitique...")

        filtered_news = news_articles
        if filter_sentiment != "Tous":
            filtered_news = [a for a in filtered_news if a["sentiment"] == filter_sentiment]
        if search_query:
            filtered_news = [a for a in filtered_news if search_query.lower() in a["title"].lower()]

        st.markdown(f"**{len(filtered_news)} articles** analysés")

        for article in filtered_news[:50]:
            polarity  = article["polarity"]
            sentiment = article["sentiment"]
            css_class = f"news-{'positive' if sentiment=='positif' else 'negative' if sentiment=='négatif' else 'neutral'}"
            icon      = "📈" if sentiment == "positif" else "📉" if sentiment == "négatif" else "➡️"
            st.markdown(f"""
            <div class="news-item {css_class}">
                <div class="news-title">{icon} <b>{article['title']}</b></div>
                <div class="news-meta">
                    📡 {article['source']} &nbsp;|&nbsp;
                    🕒 {article['published']} &nbsp;|&nbsp;
                    Polarité: {polarity:+.3f} &nbsp;|&nbsp;
                    Sentiment: <b>{sentiment.upper()}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        if not filtered_news:
            st.info("Aucun article trouvé.")

    # ══════════════════════════════════════════════
    # ══════════════════════════════════════════════
    # TAB 5: TWITTER/X INTELLIGENCE
    # ══════════════════════════════════════════════
    with tab5:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a1f2e,#0d1b2a);border:1px solid #1da1f2;
                    border-radius:12px;padding:16px 24px;margin-bottom:16px;">
            <div style="color:#1da1f2;font-size:1.3em;font-weight:bold;">🐦 Twitter/X Intelligence — Comptes Clés Finance & Géopolitique</div>
            <div style="color:#8892b0;font-size:0.88em;margin-top:6px;">
                30 comptes influents analysés via Nitter RSS (sans API key) ·
                Macro · Trading · Banques centrales · Géopolitique · Tech Finance
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Global Twitter sentiment ──
        tw_s = tw_score_data
        tw_total = tw_s.get("total", 0)
        tw_bull  = tw_s.get("bull", 0)
        tw_bear  = tw_s.get("bear", 0)
        tw_neu   = tw_s.get("neutral", 0)
        tw_sc    = tw_s.get("score", 50)
        tw_pol   = tw_s.get("weighted_polarity", 0)

        if tw_sc >= 58:    tw_color="#10b981"; tw_label="SENTIMENT POSITIF"; tw_icon="📈"
        elif tw_sc <= 42:  tw_color="#ef4444"; tw_label="SENTIMENT NÉGATIF"; tw_icon="📉"
        else:              tw_color="#3b82f6"; tw_label="SENTIMENT NEUTRE";  tw_icon="➡️"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1f2e,#16213e);
                    border:2px solid {tw_color};border-radius:12px;
                    padding:16px;text-align:center;margin:10px 0;">
            <div style="font-size:1.8em">{tw_icon}</div>
            <div style="color:{tw_color};font-size:1.5em;font-weight:bold;">{tw_label}</div>
            <div style="color:#8892b0;font-size:0.88em;margin-top:8px;">
                Score Twitter: <b style="color:#ccd6f6;font-size:1.2em">{tw_sc:.1f}/100</b> &nbsp;|&nbsp;
                Polarité pondérée: <b style="color:#fbbf24">{tw_pol:+.4f}</b> &nbsp;|&nbsp;
                Tweets analysés: <b style="color:#ccd6f6">{tw_total}</b> &nbsp;|&nbsp;
                📈 <b style="color:#10b981">{tw_bull}</b> &nbsp;
                📉 <b style="color:#ef4444">{tw_bear}</b> &nbsp;
                ➡️ <b style="color:#3b82f6">{tw_neu}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Filters ──
        tw_col1, tw_col2, tw_col3 = st.columns(3)
        with tw_col1:
            tw_filter_cat = st.selectbox("📂 Filtrer par catégorie",
                ["Toutes"] + TWITTER_CATEGORIES, key="tw_cat")
        with tw_col2:
            tw_filter_sent = st.selectbox("🎭 Filtrer par sentiment",
                ["Tous", "positif", "négatif", "neutre"], key="tw_sent")
        with tw_col3:
            tw_search = st.text_input("🔍 Rechercher", placeholder="Fed, inflation, war...", key="tw_search")

        # ── Category sentiment breakdown ──
        st.markdown('<div class="section-title">📊 Sentiment par catégorie</div>', unsafe_allow_html=True)
        by_cat = tw_s.get("by_category", {})
        if by_cat:
            cat_cols = st.columns(min(len(by_cat), 4))
            for ci, (cat_name, cat_data) in enumerate(by_cat.items()):
                with cat_cols[ci % len(cat_cols)]:
                    avg_pol = cat_data.get("avg_polarity", 0)
                    cb = cat_data["bull"]; cr = cat_data["bear"]; cn = cat_data["neutral"]
                    total_c = cb + cr + cn
                    cat_color = "#10b981" if avg_pol > 0.02 else "#ef4444" if avg_pol < -0.02 else "#3b82f6"
                    st.markdown(f"""
                    <div style="background:#1a1f2e;border:1px solid {cat_color};
                                border-radius:8px;padding:10px;text-align:center;margin:4px 0;">
                        <div style="color:#ccd6f6;font-size:0.85em;font-weight:bold">{cat_name}</div>
                        <div style="color:{cat_color};font-size:1.2em;font-weight:bold">{avg_pol:+.3f}</div>
                        <div style="color:#8892b0;font-size:0.72em">
                            📈{cb} 📉{cr} ➡️{cn} | {total_c} tweets
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Account cards ──
        st.markdown('<div class="section-title">👤 Comptes suivis & statut</div>', unsafe_allow_html=True)

        # Filter accounts
        filtered_tw = twitter_articles
        if tw_filter_cat != "Toutes":
            filtered_tw = [a for a in filtered_tw if a.get("category") == tw_filter_cat]
        if tw_filter_sent != "Tous":
            filtered_tw = [a for a in filtered_tw if a.get("sentiment") == tw_filter_sent]
        if tw_search:
            filtered_tw = [a for a in filtered_tw
                          if tw_search.lower() in a.get("title","").lower()
                          or tw_search.lower() in a.get("account_name","").lower()]

        # Group by account
        accounts_seen = {}
        for art in twitter_articles:
            handle = art.get("source", "")
            if handle not in accounts_seen:
                accounts_seen[handle] = {
                    "name": art.get("account_name",""),
                    "cat": art.get("category",""),
                    "weight": art.get("weight", 0.8),
                    "tweets": [],
                    "available": art.get("type") != "twitter_unavailable",
                }
            if art.get("type") != "twitter_unavailable":
                accounts_seen[handle]["tweets"].append(art)

        acc_cols = st.columns(3)
        for ai, (handle, acc_data) in enumerate(accounts_seen.items()):
            with acc_cols[ai % 3]:
                tweets = acc_data["tweets"]
                available = acc_data["available"]
                avg_pol = float(np.mean([t["polarity"] for t in tweets])) if tweets else 0.0
                n_bull = sum(1 for t in tweets if t["sentiment"]=="positif")
                n_bear = sum(1 for t in tweets if t["sentiment"]=="négatif")

                if not available:
                    status_color = "#4b5563"; status_icon = "⚫"; status_txt = "Non disponible"
                elif avg_pol > 0.05:
                    status_color = "#10b981"; status_icon = "🟢"; status_txt = "Positif"
                elif avg_pol < -0.05:
                    status_color = "#ef4444"; status_icon = "🔴"; status_txt = "Négatif"
                else:
                    status_color = "#3b82f6"; status_icon = "🔵"; status_txt = "Neutre"

                st.markdown(f"""
                <div style="background:#1a1f2e;border:1px solid {status_color};
                            border-radius:8px;padding:10px;margin:4px 0;min-height:90px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="color:#ccd6f6;font-weight:bold;font-size:0.88em">
                                {status_icon} {acc_data['name']}</div>
                            <div style="color:#1da1f2;font-size:0.75em">{handle}</div>
                            <div style="background:#1e3a5f;color:#93c5fd;padding:1px 6px;
                                        border-radius:8px;font-size:0.68em;display:inline-block;margin-top:2px">
                                {acc_data['cat']}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:{status_color};font-size:1.1em;font-weight:bold">
                                {avg_pol:+.3f}</div>
                            <div style="color:#8892b0;font-size:0.7em">
                                {len(tweets)} tweets<br>
                                📈{n_bull} 📉{n_bear}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Tweet feed ──
        st.markdown(f'<div class="section-title">📜 Flux tweets analysés ({len(filtered_tw)} résultats)</div>',
                    unsafe_allow_html=True)

        # Sort by absolute polarity (strongest signals first)
        sorted_tw = sorted(
            [a for a in filtered_tw if a.get("type") != "twitter_unavailable"],
            key=lambda x: abs(x.get("polarity", 0)), reverse=True
        )

        for art in sorted_tw[:60]:
            sentiment = art.get("sentiment","neutre")
            polarity  = art.get("polarity", 0)
            handle    = art.get("source","")
            name      = art.get("account_name","")
            cat       = art.get("category","")
            title     = art.get("title","")
            pub       = art.get("published","")
            weight    = art.get("weight", 0.8)
            link      = art.get("link","#")

            border_color = "#10b981" if sentiment=="positif" else "#ef4444" if sentiment=="négatif" else "#3b82f6"
            sent_icon    = "📈" if sentiment=="positif" else "📉" if sentiment=="négatif" else "➡️"

            st.markdown(f"""
            <div style="background:#1a1f2e;border-left:4px solid {border_color};
                        border-radius:0 8px 8px 0;padding:10px 14px;margin:6px 0;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div style="flex:1;">
                        <div style="color:#1da1f2;font-size:0.78em;font-weight:bold;margin-bottom:3px;">
                            {sent_icon} <a href="{link}" target="_blank"
                               style="color:#1da1f2;text-decoration:none;">
                               {handle}</a>
                            <span style="color:#8892b0"> — {name}</span>
                            <span style="background:#1e3a5f;color:#93c5fd;padding:1px 6px;
                                         border-radius:8px;font-size:0.75em;margin-left:6px">{cat}</span>
                        </div>
                        <div style="color:#ccd6f6;font-size:0.88em;">{title}</div>
                        <div style="color:#8892b0;font-size:0.72em;margin-top:4px;">
                            🕒 {pub} &nbsp;|&nbsp;
                            Polarité: <b style="color:{border_color}">{polarity:+.3f}</b> &nbsp;|&nbsp;
                            Poids: <b style="color:#fbbf24">{weight}</b> &nbsp;|&nbsp;
                            Sentiment: <b style="color:{border_color}">{sentiment.upper()}</b>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if not sorted_tw:
            st.info("Aucun tweet disponible. Les instances Nitter peuvent être temporairement hors ligne.")
            st.markdown("""
            <div style="background:#1a1f2e;border:1px solid #1e3a5f;border-radius:8px;padding:12px;margin:10px 0">
                <div style="color:#8892b0;font-size:0.85em;">
                <b style="color:#1da1f2">ℹ️ Note :</b> Les données Twitter sont récupérées via
                <b>Nitter</b> (miroirs publics sans API key). Si toutes les instances sont hors ligne,
                rafraîchissez dans quelques minutes. Les comptes suivis restent visibles ci-dessus.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Source list ──
        with st.expander("📋 Liste complète des 30 comptes suivis"):
            rows_tw = []
            for name, meta in TWITTER_ACCOUNTS.items():
                rows_tw.append({
                    "Compte": name,
                    "Handle": f"@{meta['handle']}",
                    "Catégorie": meta["cat"],
                    "Poids": f"{meta['weight']:.2f}",
                    "URL": f"https://twitter.com/{meta['handle']}",
                })
            st.dataframe(pd.DataFrame(rows_tw), use_container_width=True)


        # TAB 6: CONSEILS SCALPING  ← NEW
    # ══════════════════════════════════════════════
    with tab6:
        st.markdown('<div class="section-title">🎯 Conseils Scalping — Niveaux d\'entrée & sortie</div>', unsafe_allow_html=True)

        st.warning("⚠️ Ces conseils sont générés algorithmiquement à titre éducatif. Ne constituent pas un conseil financier.")

        # ── Sélecteurs ──
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        with col_sel1:
            scalp_market = st.selectbox(
                "📊 Actif à analyser",
                options=list(MARKETS.keys()),
                key="scalp_market"
            )
        with col_sel2:
            scalp_tf = st.selectbox(
                "⏱️ Horizon de trade",
                options=list(TIMEFRAMES.keys()),
                key="scalp_tf"
            )
        with col_sel3:
            trade_size = st.number_input(
                "💵 Taille du trade ($)",
                min_value=10.0, max_value=100000.0,
                value=100.0, step=10.0,
                key="trade_size"
            )

        # ── Compute ──
        scalp_pred  = all_predictions[scalp_market].get(scalp_tf, {})
        # Fetch fresh df for the chosen scalp timeframe (may differ from sidebar)
        scalp_tf_info = TIMEFRAMES[scalp_tf]
        scalp_df = fetch_ohlc(
            MARKETS[scalp_market]["ticker"],
            scalp_tf_info["yf_period"],
            scalp_tf_info["yf_interval"]
        )
        # Fallback to dashboard df if not enough data
        if scalp_df.empty or len(scalp_df) < 5:
            scalp_df = all_market_prices[scalp_market]["df"]
        advice = compute_scalp_advice(scalp_df, scalp_pred, scalp_market, scalp_tf, trade_size)

        if not advice:
            st.error("Données insuffisantes pour générer des conseils.")
        else:
            cp    = advice["current_price"]
            minfo = MARKETS[scalp_market]

            # ── Main advice banner ──
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1f2e, #16213e);
                        border: 2px solid {advice['advice_color']};
                        border-radius: 12px; padding: 20px; text-align: center; margin: 10px 0;">
                <div style="font-size: 2em;">{advice['advice_icon']}</div>
                <div style="color: {advice['advice_color']}; font-size: 1.6em; font-weight: bold;">
                    {advice['main_advice']}
                </div>
                <div style="color: #8892b0; font-size: 0.9em; margin-top: 8px;">
                    {minfo['icon']} <b>{scalp_market}</b> &nbsp;|&nbsp;
                    Prix actuel: <b style="color:#ccd6f6">{cp:,.4f}</b> &nbsp;|&nbsp;
                    ATR: <b style="color:#fbbf24">{advice['atr']:.4f}</b> &nbsp;|&nbsp;
                    RSI: <b style="color:#a78bfa">{advice['rsi']:.1f}</b> &nbsp;|&nbsp;
                    Horizon: <b style="color:#00d4ff">{scalp_tf}</b>
                </div>
                <div style="margin-top: 10px;">
                    <span style="color:#10b981; font-size:1.1em;">📈 Hausse: <b>{advice['bull_prob']}%</b></span>
                    &nbsp;&nbsp;&nbsp;
                    <span style="color:#ef4444; font-size:1.1em;">📉 Baisse: <b>{advice['bear_prob']}%</b></span>
                    &nbsp;&nbsp;&nbsp;
                    <span style="color:#fbbf24; font-size:1.1em;">🎯 Confiance: <b>{advice['confidence']}%</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Two columns: LONG / SHORT ──
            col_long, col_short = st.columns(2)

            # ─── LONG ───
            with col_long:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #064e3b, #065f46);
                            border: 1px solid #10b981; border-radius: 10px;
                            padding: 16px; margin-bottom: 10px; text-align:center;">
                    <div style="color:#6ee7b7; font-size:1.3em; font-weight:bold;">📈 POSITION LONGUE (ACHAT)</div>
                </div>
                """, unsafe_allow_html=True)

                def lvl_card(label, value, color, note=""):
                    return f"""
                    <div style="background:#1a1f2e; border-left: 4px solid {color};
                                padding: 10px 14px; margin: 6px 0; border-radius: 0 8px 8px 0;">
                        <div style="color:#8892b0; font-size:0.78em; text-transform:uppercase;">{label}</div>
                        <div style="color:{color}; font-size:1.3em; font-weight:bold;">{value}</div>
                        {"<div style='color:#8892b0;font-size:0.75em;'>"+note+"</div>" if note else ""}
                    </div>"""

                st.markdown(lvl_card("🟢 Entrée Long", f"{advice['entry_long']:,.4f}", "#10b981", "Légèrement au-dessus du prix actuel"), unsafe_allow_html=True)
                st.markdown(lvl_card("🔴 Stop Loss",   f"{advice['sl_long']:,.4f}",    "#ef4444", f"Risque: -{advice['loss_long']:.2f}$ sur {trade_size:.0f}$"), unsafe_allow_html=True)
                st.markdown(lvl_card("🎯 TP1 (50%)",   f"{advice['tp1_long']:,.4f}",   "#fbbf24", f"Gain estimé: +{advice['gain_long']:.2f}$"), unsafe_allow_html=True)
                st.markdown(lvl_card("🎯 TP2 (30%)",   f"{advice['tp2_long']:,.4f}",   "#f59e0b"), unsafe_allow_html=True)
                if advice["tp3_long"]:
                    st.markdown(lvl_card("🎯 TP3 (20%)", f"{advice['tp3_long']:,.4f}", "#d97706"), unsafe_allow_html=True)

                # R:R and proba
                rr_color = "#10b981" if advice["rr_long"] >= 1.5 else "#fbbf24" if advice["rr_long"] >= 1.0 else "#ef4444"
                st.markdown(f"""
                <div style="background:#16213e; border:1px solid #1e3a5f; border-radius:8px;
                            padding:12px; margin-top:10px; text-align:center;">
                    <span style="color:#8892b0; font-size:0.85em;">Ratio R:R &nbsp;</span>
                    <span style="color:{rr_color}; font-size:1.4em; font-weight:bold;">1:{advice['rr_long']}</span>
                    &nbsp;&nbsp;
                    <span style="color:#8892b0; font-size:0.85em;">Prob. gain &nbsp;</span>
                    <span style="color:#10b981; font-size:1.4em; font-weight:bold;">{advice['prob_gain_long']}%</span>
                </div>
                """, unsafe_allow_html=True)

                # P&L bar
                fig_long_pnl = go.Figure()
                fig_long_pnl.add_trace(go.Bar(
                    x=["Gain potentiel", "Perte potentielle"],
                    y=[advice["gain_long"], -advice["loss_long"]],
                    marker_color=["#10b981", "#ef4444"],
                    text=[f"+${advice['gain_long']:.2f}", f"-${advice['loss_long']:.2f}"],
                    textposition="outside",
                ))
                fig_long_pnl.update_layout(
                    template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    title=f"P&L Long sur ${trade_size:.0f}", height=220,
                    font=dict(color="#8892b0", size=11), margin=dict(t=40, b=20, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_long_pnl, use_container_width=True, config={"displayModeBar": False})

            # ─── SHORT ───
            with col_short:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #7f1d1d, #991b1b);
                            border: 1px solid #ef4444; border-radius: 10px;
                            padding: 16px; margin-bottom: 10px; text-align:center;">
                    <div style="color:#fca5a5; font-size:1.3em; font-weight:bold;">📉 POSITION COURTE (VENTE)</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(lvl_card("🔴 Entrée Short", f"{advice['entry_short']:,.4f}", "#ef4444", "Légèrement en dessous du prix actuel"), unsafe_allow_html=True)
                st.markdown(lvl_card("🟢 Stop Loss",    f"{advice['sl_short']:,.4f}",    "#10b981", f"Risque: -{advice['loss_short']:.2f}$ sur {trade_size:.0f}$"), unsafe_allow_html=True)
                st.markdown(lvl_card("🎯 TP1 (50%)",    f"{advice['tp1_short']:,.4f}",   "#fbbf24", f"Gain estimé: +{advice['gain_short']:.2f}$"), unsafe_allow_html=True)
                st.markdown(lvl_card("🎯 TP2 (30%)",    f"{advice['tp2_short']:,.4f}",   "#f59e0b"), unsafe_allow_html=True)
                if advice["tp3_short"]:
                    st.markdown(lvl_card("🎯 TP3 (20%)", f"{advice['tp3_short']:,.4f}", "#d97706"), unsafe_allow_html=True)

                rr_color_s = "#10b981" if advice["rr_short"] >= 1.5 else "#fbbf24" if advice["rr_short"] >= 1.0 else "#ef4444"
                st.markdown(f"""
                <div style="background:#16213e; border:1px solid #1e3a5f; border-radius:8px;
                            padding:12px; margin-top:10px; text-align:center;">
                    <span style="color:#8892b0; font-size:0.85em;">Ratio R:R &nbsp;</span>
                    <span style="color:{rr_color_s}; font-size:1.4em; font-weight:bold;">1:{advice['rr_short']}</span>
                    &nbsp;&nbsp;
                    <span style="color:#8892b0; font-size:0.85em;">Prob. gain &nbsp;</span>
                    <span style="color:#ef4444; font-size:1.4em; font-weight:bold;">{advice['prob_gain_short']}%</span>
                </div>
                """, unsafe_allow_html=True)

                fig_short_pnl = go.Figure()
                fig_short_pnl.add_trace(go.Bar(
                    x=["Gain potentiel", "Perte potentielle"],
                    y=[advice["gain_short"], -advice["loss_short"]],
                    marker_color=["#10b981", "#ef4444"],
                    text=[f"+${advice['gain_short']:.2f}", f"-${advice['loss_short']:.2f}"],
                    textposition="outside",
                ))
                fig_short_pnl.update_layout(
                    template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    title=f"P&L Short sur ${trade_size:.0f}", height=220,
                    font=dict(color="#8892b0", size=11), margin=dict(t=40, b=20, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_short_pnl, use_container_width=True, config={"displayModeBar": False})

            # ── Key levels summary ──
            st.markdown('<div class="section-title">📐 Niveaux clés de référence</div>', unsafe_allow_html=True)
            col_lv1, col_lv2, col_lv3, col_lv4 = st.columns(4)
            with col_lv1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Bollinger Upper</div>
                    <div class="metric-value" style="color:#fbbf24">{advice['bb_upper']:,.4f}</div>
                    <div style="color:#8892b0;font-size:0.8em">Résistance dynamique</div>
                </div>""", unsafe_allow_html=True)
            with col_lv2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Bollinger Lower</div>
                    <div class="metric-value" style="color:#fbbf24">{advice['bb_lower']:,.4f}</div>
                    <div style="color:#8892b0;font-size:0.8em">Support dynamique</div>
                </div>""", unsafe_allow_html=True)
            with col_lv3:
                ema_color = "#10b981" if cp > advice['ema9'] else "#ef4444"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">EMA 9</div>
                    <div class="metric-value" style="color:{ema_color}">{advice['ema9']:,.4f}</div>
                    <div style="color:#8892b0;font-size:0.8em">{"Prix > EMA9 ✅" if cp > advice['ema9'] else "Prix < EMA9 ⚠️"}</div>
                </div>""", unsafe_allow_html=True)
            with col_lv4:
                ema21_color = "#10b981" if cp > advice['ema21'] else "#ef4444"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">EMA 21</div>
                    <div class="metric-value" style="color:{ema21_color}">{advice['ema21']:,.4f}</div>
                    <div style="color:#8892b0;font-size:0.8em">{"Prix > EMA21 ✅" if cp > advice['ema21'] else "Prix < EMA21 ⚠️"}</div>
                </div>""", unsafe_allow_html=True)

            # ── Multi-timeframe summary for this asset ──
            st.markdown(f'<div class="section-title">🔭 Vue multi-timeframe — {scalp_market}</div>', unsafe_allow_html=True)
            tf_rows = []
            for tf_name_loop in TIMEFRAMES.keys():
                p = all_predictions[scalp_market].get(tf_name_loop, {})
                adv = compute_scalp_advice(scalp_df, p, scalp_market, tf_name_loop, trade_size)
                if adv:
                    tf_rows.append({
                        "Horizon":       tf_name_loop,
                        "Direction":     p.get("direction","—"),
                        "Prob. Hausse":  f"{p.get('bull_prob',50)}%",
                        "Prob. Baisse":  f"{p.get('bear_prob',50)}%",
                        "Confiance":     f"{p.get('confidence',0):.0f}%",
                        "Conseil":       adv.get("main_advice","—"),
                        "Entry Long":    f"{adv.get('entry_long',0):,.4f}",
                        "SL Long":       f"{adv.get('sl_long',0):,.4f}",
                        "TP1 Long":      f"{adv.get('tp1_long',0):,.4f}",
                        "R:R Long":      f"1:{adv.get('rr_long',0)}",
                        "P.Gain Long":   f"{adv.get('prob_gain_long',0)}%",
                        f"Gain+${trade_size:.0f}": f"+${adv.get('gain_long',0):.2f}",
                    })
            if tf_rows:
                st.dataframe(pd.DataFrame(tf_rows).set_index("Horizon"), use_container_width=True)

            # ── All markets quick summary ──
            st.markdown('<div class="section-title">🌐 Conseils rapides — Tous les actifs</div>', unsafe_allow_html=True)
            mkt_cols = st.columns(3)
            for mi, (mn, minfo2) in enumerate(MARKETS.items()):
                with mkt_cols[mi % 3]:
                    p_quick = all_predictions[mn].get(scalp_tf, {})
                    adv_q   = compute_scalp_advice(all_market_prices[mn]["df"], p_quick, mn, scalp_tf, trade_size)
                    if adv_q:
                        ac = adv_q["advice_color"]
                        st.markdown(f"""
                        <div style="background:#1a1f2e; border:1px solid {ac};
                                    border-radius:10px; padding:12px; margin:6px 0; text-align:center;">
                            <div style="font-size:1.3em">{minfo2['icon']}</div>
                            <div style="color:#ccd6f6; font-weight:bold; font-size:0.95em">{mn}</div>
                            <div style="color:{ac}; font-weight:bold; margin:4px 0">{adv_q['advice_icon']} {adv_q['main_advice']}</div>
                            <div style="color:#8892b0; font-size:0.78em">
                                Prix: <b style="color:#ccd6f6">{adv_q['current_price']:,.3f}</b><br>
                                Entry Long: <b style="color:#10b981">{adv_q['entry_long']:,.3f}</b> &nbsp;
                                SL: <b style="color:#ef4444">{adv_q['sl_long']:,.3f}</b><br>
                                TP1: <b style="color:#fbbf24">{adv_q['tp1_long']:,.3f}</b> &nbsp;
                                R:R <b style="color:#a78bfa">1:{adv_q['rr_long']}</b><br>
                                Prob.gain: <span style="color:#10b981"><b>{adv_q['prob_gain_long']}%</b></span> &nbsp;
                                Gain: <span style="color:#10b981"><b>+${adv_q['gain_long']:.2f}</b></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # ══════════════════════════════════════════════
    # TAB 7: SIMULATION IA  (Multi-Agent + Monte Carlo)
    # ══════════════════════════════════════════════
    with tab7:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f3460,#16213e);border:1px solid #00d4ff;
                    border-radius:12px;padding:16px 24px;margin-bottom:16px;">
            <div style="color:#00d4ff;font-size:1.3em;font-weight:bold;">🤖 Simulation Multi-Agents — MiroFish Inspired</div>
            <div style="color:#8892b0;font-size:0.88em;margin-top:6px;">
                6 agents IA analysent simultanément le marché (Macro · Technique · Sentiment · Earnings · Flow · Risk)<br>
                Monte Carlo 500 simulations · Scénarios Bull/Base/Bear · Heatmap probabilités multi-horizons
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.warning("⚠️ Simulation éducative. Pas de conseil financier.")

        # ── Controls ──
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_market = st.selectbox("📊 Actif à simuler", list(MARKETS.keys()), key="sim_market")
        with sim_col2:
            sim_tf = st.selectbox("⏱️ Horizon", list(TIMEFRAMES.keys()), key="sim_tf")
        with sim_col3:
            n_sims = st.selectbox("🎲 Simulations MC", [200, 500, 1000], index=1, key="n_sims")

        # ── Run agents ──
        sim_df = all_market_prices[sim_market]["df"]
        sim_pi = all_market_prices[sim_market]["price_info"]

        with st.spinner("🔄 Agents en cours d'analyse..."):
            agents = [
                run_agent_macro_strategist(sim_df, fund_data),
                run_agent_technical_analyst(sim_df),
                run_agent_sentiment_analyst(news_articles, fund_data),
                run_agent_earnings_analyst(sim_market, sim_pi),
                run_agent_flow_analyst(sim_df),
                run_agent_risk_analyst(sim_df, sim_market),
            ]
            tf_minutes = TIMEFRAMES[sim_tf]["minutes"]
            mc = run_monte_carlo_simulation(sim_df, agents, tf_minutes, n_sims)
            scenarios = build_scenario_matrix(agents, mc)

        # ── Agent consensus banner ──
        if mc:
            ws = mc["weighted_score"]
            bull_p = mc["bull_prob"]
            if ws >= 60:   cons_color="#10b981"; cons_label="CONSENSUS HAUSSIER"; cons_icon="📈"
            elif ws <= 40: cons_color="#ef4444"; cons_label="CONSENSUS BAISSIER"; cons_icon="📉"
            else:          cons_color="#3b82f6"; cons_label="CONSENSUS NEUTRE";   cons_icon="➡️"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1f2e,#16213e);
                        border:2px solid {cons_color};border-radius:12px;
                        padding:18px;text-align:center;margin:12px 0;">
                <div style="font-size:2em">{cons_icon}</div>
                <div style="color:{cons_color};font-size:1.7em;font-weight:bold;">{cons_label}</div>
                <div style="color:#8892b0;margin-top:8px;font-size:0.9em;">
                    Score agrégé: <b style="color:#ccd6f6;font-size:1.2em">{ws:.1f}/100</b> &nbsp;|&nbsp;
                    Hausse MC: <b style="color:#10b981">{bull_p}%</b> &nbsp;|&nbsp;
                    Baisse MC: <b style="color:#ef4444">{mc["bear_prob"]}%</b> &nbsp;|&nbsp;
                    Retour attendu: <b style="color:#fbbf24">{mc["expected_return"]:+.3f}%</b> &nbsp;|&nbsp;
                    Vol: <b style="color:#a78bfa">{mc["vol_outcome"]:.3f}%</b>
                </div>
                <div style="color:#8892b0;font-size:0.8em;margin-top:6px;">
                    {n_sims} simulations · Prix actuel: {mc["current_price"]:,.4f} · Médiane: {mc["p50"]:,.4f} · Drift/bar: {mc["drift_per_bar"]:+.4f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Agent cards ──
        st.markdown('<div class="section-title">🧠 Résultats par agent</div>', unsafe_allow_html=True)
        agent_cols = st.columns(3)
        for i, agent in enumerate(agents):
            with agent_cols[i % 3]:
                sc  = agent["score"]
                bias = agent["bias"]
                conf = agent["confidence"]
                if bias == "BULL":   bc="#10b981"; bg="rgba(16,185,129,0.1)"; bi="📈"
                elif bias == "BEAR": bc="#ef4444"; bg="rgba(239,68,68,0.1)";  bi="📉"
                else:                bc="#3b82f6"; bg="rgba(59,130,246,0.1)"; bi="➡️"

                # Top 3 signals
                sigs_html = ""
                for sig_txt, sig_val, sig_col in agent.get("signals", [])[:3]:
                    sigs_html += f'<div style="color:{sig_col};font-size:0.72em;margin:2px 0;">• {sig_txt[:55]} <b>({sig_val})</b></div>'

                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bc};border-radius:10px;
                            padding:12px;margin:5px 0;min-height:160px;">
                    <div style="color:{bc};font-weight:bold;font-size:0.95em">{agent["name"]}</div>
                    <div style="display:flex;align-items:center;margin:8px 0;">
                        <div style="font-size:1.8em;margin-right:8px">{bi}</div>
                        <div>
                            <div style="color:#ccd6f6;font-size:1.4em;font-weight:bold">{sc:.0f}/100</div>
                            <div style="color:{bc};font-size:0.8em">{bias} · conf {conf:.0f}%</div>
                        </div>
                    </div>
                    {sigs_html}
                </div>
                """, unsafe_allow_html=True)

        # ── Charts row 1: Radar + MC paths ──
        st.markdown('<div class="section-title">📊 Visualisations</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.6])
        with c1:
            if agents:
                st.plotly_chart(make_agent_radar(agents, sim_market), use_container_width=True,
                                config={"displayModeBar": False})
        with c2:
            if mc:
                st.plotly_chart(make_monte_carlo_chart(mc, sim_market), use_container_width=True,
                                config={"displayModeBar": False})

        # ── Charts row 2: Distribution + Scenarios ──
        c3, c4 = st.columns([1, 1.2])
        with c3:
            if mc:
                st.plotly_chart(make_probability_distribution(mc, sim_market),
                                use_container_width=True, config={"displayModeBar": False})
        with c4:
            if scenarios:
                st.plotly_chart(make_scenario_chart(scenarios, mc),
                                use_container_width=True, config={"displayModeBar": False})

        # ── Scenario detail cards ──
        if scenarios:
            st.markdown('<div class="section-title">🎯 Détail des scénarios</div>', unsafe_allow_html=True)
            sc_cols = st.columns(3)
            for i, sc_item in enumerate(scenarios):
                with sc_cols[i]:
                    ret = sc_item["expected_return"]
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#1a1f2e,#16213e);
                                border:2px solid {sc_item['color']};border-radius:12px;
                                padding:16px;text-align:center;">
                        <div style="font-size:1.6em">{sc_item['icon']}</div>
                        <div style="color:{sc_item['color']};font-size:1.1em;font-weight:bold;margin:4px 0">
                            {sc_item['name']}
                        </div>
                        <div style="color:#8892b0;font-size:0.8em">Score agrégé</div>
                        <div style="color:#ccd6f6;font-size:1.5em;font-weight:bold">{sc_item['score']}/100</div>
                        <div style="color:#8892b0;font-size:0.8em;margin-top:8px">Prix cible</div>
                        <div style="color:{sc_item['color']};font-size:1.3em;font-weight:bold">
                            {sc_item['target_price']:,.4f}
                        </div>
                        <div style="color:#fbbf24;font-size:0.9em">{ret:+.3f}%</div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:8px;
                                    padding:6px;margin-top:8px;">
                            <div style="color:#8892b0;font-size:0.75em">Probabilité</div>
                            <div style="color:{sc_item['color']};font-size:1.4em;font-weight:bold">
                                {sc_item['probability']}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Heatmap all markets ──
        st.markdown('<div class="section-title">🌡️ Heatmap probabilités — Tous actifs × Tous horizons</div>', unsafe_allow_html=True)
        st.plotly_chart(make_probability_heatmap(all_predictions, MARKETS),
                        use_container_width=True, config={"displayModeBar": False})

        # ── Full probability table ──
        st.markdown('<div class="section-title">📋 Tableau complet des probabilités</div>', unsafe_allow_html=True)
        prob_rows = []
        for mn in MARKETS:
            preds_mn = all_predictions.get(mn, {})
            pi_mn    = all_market_prices[mn]["price_info"]
            price_mn = pi_mn.get("price", 0)
            chg_mn   = pi_mn.get("change_pct", 0)
            row = {
                "Actif": f"{MARKETS[mn]['icon']} {mn}",
                "Prix": f"{price_mn:,.3f}" if price_mn else "—",
                "Var%": f"{chg_mn:+.2f}%",
            }
            for tf in ["5 min","15 min","60 min","240 min"]:
                p = preds_mn.get(tf, {})
                b  = p.get("bull_prob", 50)
                be = p.get("bear_prob", 50)
                d  = p.get("direction","—")
                cf = p.get("confidence",0)
                icon = "📈" if d=="HAUSSIER" else "📉" if d=="BAISSIER" else "➡️"
                row[tf] = f"{icon} {b}%↑ {be}%↓ (cf:{cf:.0f}%)"
            # Run quick MC for this market
            df_mn = all_market_prices[mn]["df"]
            if not df_mn.empty and len(df_mn) >= 10:
                agents_mn = [
                    run_agent_macro_strategist(df_mn, fund_data),
                    run_agent_technical_analyst(df_mn),
                    run_agent_sentiment_analyst(news_articles, fund_data),
                ]
                mc_mn = run_monte_carlo_simulation(df_mn, agents_mn, 60, 200)
                if mc_mn:
                    row["MC Score"] = f"{mc_mn['weighted_score']:.0f}/100"
                    row["MC Bull%"] = f"{mc_mn['bull_prob']}%"
                    row["Retour exp."] = f"{mc_mn['expected_return']:+.3f}%"
                else:
                    row["MC Score"] = "—"; row["MC Bull%"] = "—"; row["Retour exp."] = "—"
            else:
                row["MC Score"] = "—"; row["MC Bull%"] = "—"; row["Retour exp."] = "—"
            prob_rows.append(row)

        st.dataframe(pd.DataFrame(prob_rows).set_index("Actif"), use_container_width=True)

        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #1e3a5f;border-radius:8px;padding:12px;margin-top:10px;">
            <div style="color:#8892b0;font-size:0.8em;">
            <b style="color:#00d4ff">Méthodologie :</b>
            6 agents IA (Macro Strategist · Technical Analyst · Sentiment Analyst · Earnings Analyst · Flow Analyst · Risk Analyst)
            analysent indépendamment le marché. Leurs scores sont pondérés (Tech 35% · Macro 20% · Earnings 15% · Sentiment 15% · Flow 10% · Risk 5%)
            pour produire un consensus. Le Monte Carlo simule N chemins de prix à partir de la volatilité historique et du drift calculé.
            Les scénarios Bull/Base/Bear ajustent ce consensus de ±15 points.
            </div>
        </div>
        """, unsafe_allow_html=True)

    
        # ══════════════════════════════════════════════════════
    # TAB 8: MIROFISH LAB
    # ══════════════════════════════════════════════════════
    with tab8:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0a0f1e,#111827);
                    border:1px solid #6366f1;border-radius:12px;padding:16px 24px;margin-bottom:12px;">
            <div style="color:#818cf8;font-size:1.4em;font-weight:bold;">
                🧪 MiroFish Swarm Intelligence Lab
            </div>
            <div style="color:#8892b0;font-size:0.85em;margin-top:6px;">
                Inspiré du quant chinois (18k ⭐ GitHub) · 40 ans historique SPX · 4 agents swarm ·
                VADER NLP · Monte Carlo 1000 sims · Reddit + Google News + Twitter/X ·
                Rapport PDF 1 page pro à télécharger
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.warning("⚠️ Simulation éducative. Les probabilités ne garantissent pas les résultats.")

        # ── Controls ──
        mf_c1, mf_c2, mf_c3, mf_c4 = st.columns(4)
        with mf_c1:
            mf_asset = st.selectbox("📊 Actif", list(MARKETS.keys()), key="mf_asset")
        with mf_c2:
            mf_tf = st.selectbox("⏱️ Timeframe", list(TIMEFRAMES.keys()), key="mf_tf")
        with mf_c3:
            mf_sims = st.selectbox("🎲 Simulations", [500, 1000, 2000], index=1, key="mf_sims")
        with mf_c4:
            mf_run = st.button("🚀 LANCER MIROFISH", use_container_width=True,
                               type="primary", key="mf_run")

        # Custom tweet input
        with st.expander("📝 Coller vos tweets / textes manuels (optionnel — enrichit l'analyse)"):
            st.markdown("""
            **Format :** Collez directement des tweets, titres d'articles, ou tout texte financier.
            L'analyse VADER s'applique automatiquement.
            ```
            ANALYSE MAINTENANT : [collez ici vos tweets ou CSV]
            Actif : Nasdaq/$QQQ
            Timeframe : 5min
            ```
            """)
            custom_input = st.text_area(
                "Textes personnalisés (un par ligne ou bloc libre)",
                height=120,
                placeholder="Fed signals rate cut... Oil surges on OPEC cut... Nasdaq futures up 0.5%...",
                key="mf_custom"
            )

        if mf_run or st.session_state.get("mf_result"):

            mf_df         = all_market_prices[mf_asset]["df"]
            mf_tf_minutes = TIMEFRAMES[mf_tf]["minutes"]

            with st.spinner("🧠 MiroFish Swarm en cours — VADER · 4 agents · 1000 MC..."):
                # Gather all texts
                all_mf_texts = []

                # Twitter
                for a in twitter_articles:
                    if a.get("type") != "twitter_unavailable" and a.get("title"):
                        all_mf_texts.append(a["title"])

                # RSS News
                for a in news_articles:
                    if a.get("title"):
                        all_mf_texts.append(a["title"])

                # Reddit
                for a in reddit_articles:
                    if a.get("title"):
                        all_mf_texts.append(a["title"])

                # Google News
                for a in gnews_articles:
                    if a.get("title"):
                        all_mf_texts.append(a["title"])

                # Custom input
                if custom_input.strip():
                    lines = [l.strip() for l in custom_input.strip().split("\n") if len(l.strip()) > 5]
                    all_mf_texts.extend(lines)

                # Remove duplicates
                all_mf_texts = list(dict.fromkeys(all_mf_texts))

                mf_result = mirofish_swarm_simulation(
                    df=mf_df,
                    all_texts=all_mf_texts,
                    fund_score=fund_data,
                    tw_score=tw_score_data,
                    asset_name=mf_asset,
                    timeframe_minutes=mf_tf_minutes,
                    n_simulations=mf_sims,
                )
                st.session_state["mf_result"] = mf_result
                st.session_state["mf_asset_name"] = mf_asset
                st.session_state["mf_tf"] = mf_tf

            mf_result = st.session_state.get("mf_result", {})
            mf_asset_name = st.session_state.get("mf_asset_name", mf_asset)
            mf_tf_display = st.session_state.get("mf_tf", mf_tf)

            if not mf_result:
                st.error("Données insuffisantes pour lancer la simulation.")
            else:
                verdict = mf_result["vader_verdict"]
                if verdict == "HAUSSIER":   v_color="#10b981"; v_icon="📈"
                elif verdict == "BAISSIER": v_color="#ef4444"; v_icon="📉"
                else:                       v_color="#3b82f6"; v_icon="➡️"

                # ── VADER Banner ──
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a1f2e,#0d1b2a);
                            border:2px solid {v_color};border-radius:12px;
                            padding:16px;text-align:center;margin:10px 0;">
                    <div style="font-size:2em">{v_icon}</div>
                    <div style="color:{v_color};font-size:1.8em;font-weight:bold;">{verdict}</div>
                    <div style="color:#fbbf24;font-size:1.4em;font-weight:bold;">
                        VADER compound: {mf_result["vader_mean"]:+.4f}
                    </div>
                    <div style="color:#8892b0;font-size:0.85em;margin-top:6px;">
                        Règle: >+0.10=HAUSSIER · <-0.10=BAISSIER · Textes analysés: <b style="color:#ccd6f6">{mf_result["n_texts"]}</b>
                        (Twitter · Reddit · Google News · RSS · Manuel) &nbsp;|&nbsp;
                        📈{mf_result["vader_positive"]} 📉{mf_result["vader_negative"]} ➡️{mf_result["vader_neutral"]}
                    </div>
                    <div style="margin-top:10px;">
                        <span style="color:#10b981;font-size:1.1em">📈 Hausse MC: <b>{mf_result["bull_prob"]}%</b></span>
                        &nbsp;&nbsp;
                        <span style="color:#ef4444;font-size:1.1em">📉 Baisse MC: <b>{mf_result["bear_prob"]}%</b></span>
                        &nbsp;&nbsp;
                        <span style="color:#fbbf24;font-size:1.1em">📊 Consensus: <b>{mf_result["consensus"]:.1f}/100</b></span>
                        &nbsp;&nbsp;
                        <span style="color:#a78bfa;font-size:1.1em">📉 Retour exp.: <b>{mf_result["exp_return"]:+.3f}%</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── 3 cols: Themes · Agents · Trade ──
                th_col, ag_col, tr_col = st.columns([1.2, 1.5, 1])

                with th_col:
                    st.markdown('<div class="section-title">🔥 Top 3 Thèmes</div>', unsafe_allow_html=True)
                    themes = mf_result.get("themes", [])
                    if themes:
                        for i, th in enumerate(themes[:3], 1):
                            imp = th["avg_impact"]
                            tc  = "#10b981" if imp > 0 else "#ef4444"
                            st.markdown(f"""
                            <div style="background:#1a1f2e;border-left:4px solid {tc};
                                        padding:8px 12px;margin:5px 0;border-radius:0 8px 8px 0;">
                                <div style="color:#ccd6f6;font-weight:bold;font-size:0.88em">
                                    #{i} {th["label"]}</div>
                                <div style="color:#8892b0;font-size:0.75em">
                                    {th["count"]} mentions</div>
                                <div style="color:{tc};font-size:0.78em;margin-top:3px;">
                                    → {th["impact_desc"]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun thème dominant détecté.")

                with ag_col:
                    st.markdown('<div class="section-title">🤖 4 Agents Swarm</div>', unsafe_allow_html=True)
                    for agent_name, agent_data in mf_result.get("agents", {}).items():
                        sc = agent_data["score"]
                        ac = "#10b981" if sc>=60 else "#ef4444" if sc<=40 else "#3b82f6"
                        bar_w = int(sc)
                        st.markdown(f"""
                        <div style="background:#1a1f2e;border:1px solid {ac};
                                    border-radius:8px;padding:8px 10px;margin:4px 0;">
                            <div style="display:flex;justify-content:space-between;">
                                <span style="color:#ccd6f6;font-size:0.82em;font-weight:bold">{agent_name}</span>
                                <span style="color:{ac};font-size:0.9em;font-weight:bold">{sc:.0f}/100</span>
                            </div>
                            <div style="background:#0e1117;border-radius:4px;height:5px;margin:4px 0;">
                                <div style="background:{ac};width:{bar_w}%;height:5px;border-radius:4px;"></div>
                            </div>
                            <div style="color:#8892b0;font-size:0.7em">{agent_data["basis"][:70]}</div>
                        </div>
                        """, unsafe_allow_html=True)

                with tr_col:
                    st.markdown('<div class="section-title">🎯 Trade Signal</div>', unsafe_allow_html=True)
                    sig   = mf_result["trade_signal"]
                    conf  = mf_result["trade_conf"]
                    tgt   = mf_result["target_price"]
                    sl    = mf_result["sl_price"]
                    tc2   = mf_result["trade_color"]
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#1a1f2e,#16213e);
                                border:2px solid {tc2};border-radius:12px;
                                padding:14px;text-align:center;">
                        <div style="color:{tc2};font-size:1.8em;font-weight:bold">{sig}</div>
                        <div style="color:#fbbf24;font-size:1.2em;margin:6px 0">
                            Confiance: <b>{conf}%</b></div>
                        <div style="color:#8892b0;font-size:0.8em;margin-top:8px;">
                            <div>🎯 Cible: <b style="color:{tc2}">{tgt:,.4f}</b></div>
                            <div>🛑 SL: <b style="color:#ef4444">{sl:,.4f}</b></div>
                            <div style="margin-top:6px">
                                Actuel: <b style="color:#ccd6f6">{mf_result["current_price"]:,.4f}</b>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── MC Chart ──
                st.plotly_chart(
                    make_mirofish_mc_chart(mf_result, mf_asset_name, mf_tf_display),
                    use_container_width=True, config={"displayModeBar": False}
                )

                # ── Probability table ──
                st.markdown('<div class="section-title">📊 Tableau des probabilités MiroFish</div>', unsafe_allow_html=True)
                pt = mf_result["prob_table"]

                prob_html = """
                <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:10px 0;">
                """
                for move, prob in pt.items():
                    pc = "#10b981" if move.startswith("+") else "#ef4444"
                    bg = "rgba(16,185,129,0.1)" if move.startswith("+") else "rgba(239,68,68,0.1)"
                    prob_html += f"""
                    <div style="background:{bg};border:1px solid {pc};border-radius:8px;
                                padding:12px;text-align:center;">
                        <div style="color:{pc};font-size:1.1em;font-weight:bold">{move}</div>
                        <div style="color:#ccd6f6;font-size:1.5em;font-weight:bold">{prob}%</div>
                        <div style="color:#8892b0;font-size:0.7em">probabilité</div>
                    </div>"""
                prob_html += "</div>"
                st.markdown(prob_html, unsafe_allow_html=True)

                # Price levels
                c_p10, c_p50, c_p90 = st.columns(3)
                with c_p10:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Scénario pessimiste P10</div>
                        <div class="metric-value" style="color:#ef4444">{mf_result["p10"]:,.4f}</div>
                        <div style="color:#8892b0;font-size:0.75em">10% des simulations sous ce niveau</div>
                    </div>""", unsafe_allow_html=True)
                with c_p50:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Scénario médian P50</div>
                        <div class="metric-value" style="color:#fbbf24">{mf_result["p50"]:,.4f}</div>
                        <div style="color:#8892b0;font-size:0.75em">Prix médian attendu</div>
                    </div>""", unsafe_allow_html=True)
                with c_p90:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Scénario optimiste P90</div>
                        <div class="metric-value" style="color:#10b981">{mf_result["p90"]:,.4f}</div>
                        <div style="color:#8892b0;font-size:0.75em">90% des simulations sous ce niveau</div>
                    </div>""", unsafe_allow_html=True)

                # Sources breakdown
                with st.expander(f"📡 Sources utilisées — {len(all_mf_texts)} textes analysés"):
                    src_counts = {
                        "Twitter/X":    sum(1 for a in twitter_articles if a.get("type")!="twitter_unavailable"),
                        "RSS News":     len(news_articles),
                        "Reddit":       len(reddit_articles),
                        "Google News":  len(gnews_articles),
                        "Manuel":       len([l for l in custom_input.strip().split("\n") if len(l.strip())>5]) if custom_input.strip() else 0,
                    }
                    src_df = pd.DataFrame(
                        [{"Source": k, "Textes": v, "% du total": f"{v/max(len(all_mf_texts),1)*100:.1f}%"}
                         for k, v in src_counts.items() if v > 0]
                    )
                    st.dataframe(src_df, use_container_width=True)

                # ── PDF Download ──
                st.markdown("---")
                st.markdown('<div class="section-title">📄 Rapport PDF 1 page — Style Trader Pro</div>', unsafe_allow_html=True)
                dl_col1, dl_col2 = st.columns([3, 1])
                with dl_col1:
                    st.markdown("""
                    Rapport **1 page A4** complet :
                    Sentiment VADER · Top 3 thèmes · Swarm 4 agents · Monte Carlo probas ·
                    Verdict trade (LONG/SHORT/WAIT) · Commentaire pro 4 lignes · Date/heure Paris
                    """)
                with dl_col2:
                    if st.button("📥 Télécharger rapport PDF", use_container_width=True,
                                 type="primary", key="mf_pdf"):
                        with st.spinner("Génération rapport..."):
                            try:
                                pdf_bytes = generate_mirofish_pdf(
                                    mf_result, mf_asset_name, mf_tf_display, custom_input
                                )
                                b64 = base64.b64encode(pdf_bytes).decode()
                                now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                fname  = f"MiroFish_{mf_asset_name.replace(' ','_')}_{mf_tf_display.replace(' ','')}_{now_ts}.pdf"
                                href   = (
                                    f'<a href="data:application/pdf;base64,{b64}" download="{fname}" '
                                    f'style="background:linear-gradient(135deg,#4f46e5,#6366f1);color:white;'
                                    f'padding:12px 20px;border-radius:8px;text-decoration:none;'
                                    f'font-weight:bold;display:inline-block;">⬇️ {fname}</a>'
                                )
                                st.markdown(href, unsafe_allow_html=True)
                                st.success("✅ Rapport généré!")
                            except Exception as e:
                                st.error(f"Erreur PDF: {e}")
        else:
            st.markdown("""
            <div style="background:#1a1f2e;border:2px dashed #4f46e5;border-radius:12px;
                        padding:30px;text-align:center;margin:20px 0;">
                <div style="font-size:2em">🧪</div>
                <div style="color:#818cf8;font-size:1.2em;font-weight:bold;margin:8px 0">
                    Cliquez sur 🚀 LANCER MIROFISH pour démarrer la simulation</div>
                <div style="color:#8892b0;font-size:0.85em;">
                    Sélectionnez l'actif, le timeframe, collez vos tweets (optionnel)<br>
                    puis lancez l'analyse VADER + 4 agents + Monte Carlo 1000 sims
                </div>
            </div>
            """, unsafe_allow_html=True)


        # TAB 9: SOURCES
    # ══════════════════════════════════════════════
    with tab9:
        st.markdown('<div class="section-title">📋 Sources & Méthodologie</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### 📡 Sources de données de marché")
            st.markdown("""
            | Source | Type | Mise à jour |
            |--------|------|-------------|
            | **Yahoo Finance** | OHLCV temps réel | 1-5 min |
            | **yfinance API** | Historical data | Temps réel |
            | Tickers | ^NDX, ^GSPC, GC=F, ^FCHI, ^STOXX, CL=F | — |
            """)
            st.markdown("#### 📰 Flux RSS d'actualités")
            for src, url in RSS_FEEDS.items():
                st.markdown(f"- **{src}** — `{url[:55]}...`")
        with col_s2:
            st.markdown("#### 🔬 Indicateurs techniques")
            st.markdown("""
            | Indicateur | Paramètres | Usage |
            |-----------|-----------|-------|
            | EMA | 9, 21 | Tendance CT |
            | SMA | 50, 200 | Tendance LT |
            | RSI | 14 | Surachat/survente |
            | MACD | 12, 26, 9 | Momentum |
            | Bollinger | 20p, 2σ | Volatilité + niveaux |
            | Stochastique | 14, 3 | Retournements |
            | ATR | 14 | Volatilité + SL/TP |
            | OBV | — | Volume |
            """)
        st.markdown("#### 🌍 Indicateurs macro suivis (20 facteurs)")
        for name, data in ECONOMIC_INDICATORS.items():
            icon = "🟢" if data["impact"] == "positif" else "🔴" if data["impact"] in ("négatif","negatif") else "⚪"
            trend_disp = data['trend']
            st.markdown(f"- {icon} **{name}** — `{data['value']}` {trend_disp} — *{data['impact']}* (poids: {data['weight']*100:.0f}%)")
        st.markdown("#### 🐦 Comptes Twitter/X suivis (30 comptes)")
        tw_cats_grp = {}
        for name, meta in TWITTER_ACCOUNTS.items():
            c = meta["cat"]
            tw_cats_grp.setdefault(c, []).append(f"@{meta['handle']} ({name})")
        for cat, handles in sorted(tw_cats_grp.items()):
            st.markdown(f"**{cat}**: " + " · ".join(handles))
        st.warning("⚠️ **Avertissement légal** : Application éducative uniquement. Pas de conseil financier.")

    # ── PDF Report ──
    st.markdown("---")
    st.markdown('<div class="section-title">📄 Génération de rapport PDF</div>', unsafe_allow_html=True)
    col_pdf1, col_pdf2 = st.columns([2, 1])
    with col_pdf1:
        st.markdown("Rapport PDF complet : analyses, prédictions, conseils scalping, indicateurs et sources.")
    with col_pdf2:
        if st.button("📥 Générer rapport PDF", use_container_width=True, type="primary"):
            with st.spinner("Génération du rapport..."):
                try:
                    # Build scalp_advices for all markets x all timeframes
                    scalp_advices_pdf = {}
                    for mn, minfo_pdf in MARKETS.items():
                        scalp_advices_pdf[mn] = {}
                        df_pdf = all_market_prices[mn]["df"]
                        for tf_pdf in TIMEFRAMES.keys():
                            p_pdf = all_predictions[mn].get(tf_pdf, {})
                            scalp_advices_pdf[mn][tf_pdf] = compute_scalp_advice(
                                df_pdf, p_pdf, mn, tf_pdf, 100.0
                            )
                    pdf_bytes = generate_pdf_report(
                        all_market_prices,
                        all_predictions,
                        fund_data,
                        news_articles=news_articles,
                        scalp_advices=scalp_advices_pdf,
                    )
                    b64 = base64.b64encode(pdf_bytes).decode()
                    filename = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    href = (
                        f'<a href="data:application/pdf;base64,{b64}" download="{filename}" '
                        f'style="background: linear-gradient(135deg, #0f3460, #00d4ff); color: white; '
                        f'padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: bold; '
                        f'display: inline-block; margin-top: 10px;">⬇️ Télécharger le rapport PDF</a>'
                    )
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ Rapport généré avec succès!")
                except Exception as e:
                    st.error(f"Erreur PDF: {str(e)}")


if __name__ == "__main__":
    main()
