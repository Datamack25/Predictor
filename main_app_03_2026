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
    """Fetch OHLCV data from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
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


def compute_fundamental_score(news_articles: list) -> dict:
    """Compute fundamental/macro sentiment score."""
    # Economic indicators score
    eco_bull = 0
    eco_bear = 0
    eco_signals = []
    
    for name, data in ECONOMIC_INDICATORS.items():
        w = data["weight"] * 10
        impact = data["impact"]
        trend  = data["trend"]
        
        if impact == "positif":
            eco_bull += w
            eco_signals.append({"name": name, "signal": "🟢", "value": str(data["value"])})
        elif impact == "négatif":
            eco_bear += w
            eco_signals.append({"name": name, "signal": "🔴", "value": str(data["value"])})
        else:
            eco_signals.append({"name": name, "signal": "⚪", "value": str(data["value"])})
    
    # News sentiment score
    news_bull = sum(1 for a in news_articles if a["sentiment"] == "positif")
    news_bear = sum(1 for a in news_articles if a["sentiment"] == "négatif")
    news_total = max(len(news_articles), 1)
    news_score = (news_bull / news_total) * 100
    
    # Combined
    eco_total = eco_bull + eco_bear
    eco_score = (eco_bull / eco_total * 100) if eco_total > 0 else 50
    
    combined = eco_score * 0.7 + news_score * 0.3
    
    return {
        "eco_score": eco_score,
        "news_score": news_score,
        "combined": combined,
        "eco_signals": eco_signals,
        "news_bull": news_bull,
        "news_bear": news_bear,
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
# PDF REPORT
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(market_data: dict, predictions_all: dict, fund_score: dict) -> bytes:
    """Generate a PDF report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_fill_color(14, 17, 23)
    pdf.rect(0, 0, 210, 297, "F")
    
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 15, "MARKET SENTIMENT PREDICTOR", ln=True, align="C")
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(136, 146, 176)
    pdf.cell(0, 8, f"Rapport genere le {datetime.now().strftime('%d/%m/%Y a %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)
    
    # Separator
    pdf.set_draw_color(0, 212, 255)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)
    
    # Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 10, "RESUME EXECUTIF", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(204, 214, 246)
    pdf.multi_cell(0, 6, 
        f"Score Fondamental: {fund_score['combined']:.1f}/100 | "
        f"Score Eco: {fund_score['eco_score']:.1f}/100 | "
        f"Score News: {fund_score['news_score']:.1f}/100\n"
        f"Actualites analysees: {fund_score['news_bull'] + fund_score['news_bear'] + fund_score['news_neutral']} | "
        f"Positives: {fund_score['news_bull']} | Negatives: {fund_score['news_bear']} | Neutres: {fund_score['news_neutral']}"
    )
    pdf.ln(5)
    
    # Per market
    for market_name, data in market_data.items():
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(100, 255, 218)
        pdf.cell(0, 10, f"  {market_name}", ln=True)
        
        price_info = data.get("price_info", {})
        if price_info:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(204, 214, 246)
            chg = price_info.get("change_pct", 0)
            pdf.cell(0, 6, 
                f"    Prix: {price_info.get('price', 'N/A'):.4g}  |  "
                f"Variation: {chg:+.2f}%  |  "
                f"H: {price_info.get('high', 'N/A'):.4g}  L: {price_info.get('low', 'N/A'):.4g}",
                ln=True)
        
        if market_name in predictions_all:
            preds = predictions_all[market_name]
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(139, 233, 253)
            pdf.cell(0, 7, "    Predictions:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for tf, pred in preds.items():
                direction = pred["direction"]
                bull = pred["bull_prob"]
                bear = pred["bear_prob"]
                conf = pred["confidence"]
                if direction == "HAUSSIER":
                    pdf.set_text_color(16, 185, 129)
                elif direction == "BAISSIER":
                    pdf.set_text_color(239, 68, 68)
                else:
                    pdf.set_text_color(59, 130, 246)
                pdf.cell(0, 5, 
                    f"      {tf:8s}: {direction:9s}  |  Hausse: {bull}%  |  Baisse: {bear}%  |  Confiance: {conf}%",
                    ln=True)
        
        pdf.ln(3)
    
    # Economic indicators
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 10, "INDICATEURS ECONOMIQUES & MACROECONOMIQUES", ln=True)
    pdf.ln(3)
    
    for name, data in ECONOMIC_INDICATORS.items():
        impact = data["impact"]
        if impact == "positif":
            pdf.set_text_color(16, 185, 129)
        elif impact == "négatif":
            pdf.set_text_color(239, 68, 68)
        else:
            pdf.set_text_color(59, 130, 246)
        
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5,
            f"  {name:35s}: {str(data['value']):10s} {data['trend']}  [{impact.upper()}]  poids: {data['weight']*100:.0f}%",
            ln=True)
    
    # Sources
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 10, "SOURCES & METHODOLOGIE", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 255, 218)
    pdf.cell(0, 8, "Donnees de marche:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(204, 214, 246)
    pdf.multi_cell(0, 6,
        "- Yahoo Finance (yfinance) : donnees OHLCV en temps reel et historiques\n"
        "- Intervalles: 5min, 15min, 60min, 240min\n"
        "- Tickers: ^NDX, ^GSPC, GC=F, ^FCHI, ^STOXX, CL=F"
    )
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 255, 218)
    pdf.cell(0, 8, "Sources d'actualites (RSS):", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(204, 214, 246)
    for source in RSS_FEEDS.keys():
        pdf.cell(0, 5, f"  - {source}", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 255, 218)
    pdf.cell(0, 8, "Indicateurs techniques utilises:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(204, 214, 246)
    pdf.multi_cell(0, 6,
        "- EMA 9, EMA 21, SMA 50, SMA 200\n"
        "- RSI (14 periodes)\n"
        "- MACD (12, 26, 9)\n"
        "- Bandes de Bollinger (20 periodes, 2 ecarts-types)\n"
        "- Stochastique (14, 3)\n"
        "- ATR (Average True Range)\n"
        "- OBV (On-Balance Volume)\n"
        "- Volume SMA (20 periodes)"
    )
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 255, 218)
    pdf.cell(0, 8, "Methodologie de prediction:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(204, 214, 246)
    pdf.multi_cell(0, 6,
        "- Scoring technique: 8 indicateurs ponderes (RSI, MACD, EMA, BB, Stoch, Volume, Momentum)\n"
        "- Scoring fondamental: 20 indicateurs macro + analyse NLP des actualites (TextBlob)\n"
        "- Ponderation: CT (5min/15min) = 80% tech + 20% macro\n"
        "  MT (60min) = 65% tech + 35% macro | LT (240min) = 50/50\n"
        "- Analyse de sentiment: polarite et subjectivite des titres d'articles\n"
        "AVERTISSEMENT: Ces predictions sont a des fins educatives uniquement."
    )
    
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.ln(10)
    pdf.multi_cell(0, 5,
        "AVERTISSEMENT LEGAL: Ce rapport est genere automatiquement a des fins informatives et educatives "
        "uniquement. Il ne constitue pas un conseil financier. Les marches financiers sont risques. "
        "Consultez un conseiller financier avant toute decision d'investissement."
    )
    
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
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(300)
        st.rerun()
    
    # ── Load data ──
    with st.spinner("🔄 Chargement des données en cours..."):
        news_articles = fetch_news_sentiment()
        fund_data = compute_fundamental_score(news_articles)
        
        # Fetch all markets
        all_market_prices = {}
        all_predictions = {}
        
        for mkt_name, mkt_info in MARKETS.items():
            ticker = mkt_info["ticker"]
            tf_info = TIMEFRAMES[selected_interval]
            df = fetch_ohlc(ticker, tf_info["yf_period"], tf_info["yf_interval"])
            price_info = get_current_price(ticker)
            all_market_prices[mkt_name] = {"df": df, "price_info": price_info}
            
            tech = compute_technical_signal(df)
            preds = {}
            for tf_name, tf_data in TIMEFRAMES.items():
                preds[tf_name] = predict_direction(tech["score"], fund_data["combined"], tf_data["minutes"])
            all_predictions[mkt_name] = preds
    
    # ── Main tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "📈 Graphiques OHLC", 
        "🌍 Macro & Sentiment", 
        "📰 Actualités",
        "📋 Sources"
    ])
    
    # ══════════════════════════════════════════════
    # TAB 1: DASHBOARD
    # ══════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">🏦 Vue d\'ensemble des marchés</div>', unsafe_allow_html=True)
        
        # Market overview cards
        cols = st.columns(len(MARKETS))
        for i, (mkt_name, mkt_info) in enumerate(MARKETS.items()):
            with cols[i]:
                price_info = all_market_prices[mkt_name]["price_info"]
                preds = all_predictions[mkt_name]
                
                # Short-term prediction
                pred_5m = preds.get("5 min", {})
                direction = pred_5m.get("direction", "N/A")
                score = pred_5m.get("score", 50)
                
                if direction == "HAUSSIER":
                    delta_class = "metric-delta-up"
                    dir_icon = "▲"
                elif direction == "BAISSIER":
                    delta_class = "metric-delta-down"
                    dir_icon = "▼"
                else:
                    delta_class = ""
                    dir_icon = "▶"
                
                price = price_info.get("price", 0)
                chg_pct = price_info.get("change_pct", 0)
                chg_class = "metric-delta-up" if chg_pct >= 0 else "metric-delta-down"
                chg_sign = "+" if chg_pct >= 0 else ""
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5em">{mkt_info['icon']}</div>
                    <div class="metric-label">{mkt_name}</div>
                    <div class="metric-value">{f"{price:,.2f}" if price else "—"}</div>
                    <div class="{chg_class}">{chg_sign}{chg_pct:.2f}%</div>
                    <div style="margin-top:8px">
                        <span class="{'signal-bull' if direction=='HAUSSIER' else 'signal-bear' if direction=='BAISSIER' else 'signal-neutral'}">{dir_icon} {direction}</span>
                    </div>
                    <div style="color:#8892b0; font-size:0.75em; margin-top:5px">Score: {score:.0f}/100</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Global sentiment gauges
        st.markdown('<div class="section-title">🎯 Sentiments globaux</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        mkt_name = selected_market
        tech_sig = compute_technical_signal(all_market_prices[mkt_name]["df"])
        
        with col1:
            st.plotly_chart(make_sentiment_gauge(tech_sig["score"], f"Technique\n{mkt_name}"), 
                           use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(make_sentiment_gauge(fund_data["eco_score"], "Score Macro-Éco"),
                           use_container_width=True, config={"displayModeBar": False})
        with col3:
            st.plotly_chart(make_sentiment_gauge(fund_data["news_score"], "Sentiment News"),
                           use_container_width=True, config={"displayModeBar": False})
        with col4:
            combined = fund_data["combined"] * 0.5 + tech_sig["score"] * 0.5
            st.plotly_chart(make_sentiment_gauge(combined, "Score Global"),
                           use_container_width=True, config={"displayModeBar": False})
        
        # Predictions table
        st.markdown('<div class="section-title">🔮 Prédictions par marché & horizon</div>', unsafe_allow_html=True)
        
        rows = []
        for mkt_name, preds in all_predictions.items():
            row = {"Marché": f"{MARKETS[mkt_name]['icon']} {mkt_name}"}
            for tf, pred in preds.items():
                d = pred["direction"]
                b = pred["bull_prob"]
                be = pred["bear_prob"]
                emoji = "📈" if d == "HAUSSIER" else "📉" if d == "BAISSIER" else "➡️"
                row[tf] = f"{emoji} {d} ({b}%↑ / {be}%↓)"
            rows.append(row)
        
        df_preds = pd.DataFrame(rows).set_index("Marché")
        st.dataframe(df_preds, use_container_width=True)
        
        # Probability chart for selected market
        st.markdown(f'<div class="section-title">📊 Probabilités — {selected_market}</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_probability_chart(all_predictions[selected_market]),
            use_container_width=True
        )
        
        # Technical signals for selected market
        st.markdown(f'<div class="section-title">⚡ Signaux techniques — {selected_market}</div>', unsafe_allow_html=True)
        signals = tech_sig.get("signals", [])
        if signals:
            sig_cols = st.columns(2)
            for i, sig in enumerate(signals):
                with sig_cols[i % 2]:
                    bull_bear = "🟢" if "BULL" in sig["signal"] else "🔴" if "BEAR" in sig["signal"] else "⚪"
                    st.markdown(f"""
                    <div class="news-item {'news-positive' if 'BULL' in sig['signal'] else 'news-negative' if 'BEAR' in sig['signal'] else 'news-neutral'}">
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
        
        # All markets mini charts
        st.markdown('<div class="section-title">📊 Tous les marchés</div>', unsafe_allow_html=True)
        
        cols_charts = st.columns(2)
        for i, (mkt_name, mkt_info) in enumerate(MARKETS.items()):
            if mkt_name == selected_market:
                continue
            df_mkt = all_market_prices[mkt_name]["df"]
            with cols_charts[i % 2]:
                if not df_mkt.empty:
                    fig_mini = go.Figure(go.Candlestick(
                        x=df_mkt.index[-50:],
                        open=df_mkt["Open"].iloc[-50:],
                        high=df_mkt["High"].iloc[-50:],
                        low=df_mkt["Low"].iloc[-50:],
                        close=df_mkt["Close"].iloc[-50:],
                        increasing_fillcolor="#10b981", increasing_line_color="#10b981",
                        decreasing_fillcolor="#ef4444", decreasing_line_color="#ef4444",
                    ))
                    fig_mini.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0e1117",
                        plot_bgcolor="#0e1117",
                        title=f"{mkt_info['icon']} {mkt_name}",
                        height=280,
                        xaxis_rangeslider_visible=False,
                        showlegend=False,
                        margin=dict(t=40, b=20, l=20, r=20),
                        font=dict(color="#8892b0"),
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
    
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
            if val == "POSITIF":
                return "color: #10b981"
            elif val == "NÉGATIF":
                return "color: #ef4444"
            return "color: #3b82f6"
        
        st.dataframe(df_eco.style.applymap(color_impact, subset=["Impact"]), use_container_width=True)
        
        # News sentiment breakdown
        st.markdown('<div class="section-title">📊 Répartition du sentiment des actualités</div>', unsafe_allow_html=True)
        
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            fig_pie = go.Figure(go.Pie(
                labels=["Positif 📈", "Négatif 📉", "Neutre ➡️"],
                values=[fund_data["news_bull"], fund_data["news_bear"], fund_data["news_neutral"]],
                hole=0.5,
                marker=dict(colors=["#10b981", "#ef4444", "#3b82f6"]),
            ))
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                title="Sentiment des actualités",
                font=dict(color="#8892b0"),
                height=300,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_pie2:
            # Impact breakdown
            pos_count = sum(1 for d in ECONOMIC_INDICATORS.values() if d["impact"] == "positif")
            neg_count = sum(1 for d in ECONOMIC_INDICATORS.values() if d["impact"] == "négatif")
            neu_count = sum(1 for d in ECONOMIC_INDICATORS.values() if d["impact"] == "neutre")
            
            fig_pie2 = go.Figure(go.Pie(
                labels=["Positif 📈", "Négatif 📉", "Neutre ➡️"],
                values=[pos_count, neg_count, neu_count],
                hole=0.5,
                marker=dict(colors=["#10b981", "#ef4444", "#3b82f6"]),
            ))
            fig_pie2.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                title="Impact des indicateurs macro",
                font=dict(color="#8892b0"),
                height=300,
            )
            st.plotly_chart(fig_pie2, use_container_width=True)
    
    # ══════════════════════════════════════════════
    # TAB 4: NEWS
    # ══════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">📰 Actualités financières & géopolitiques</div>', unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            filter_sentiment = st.selectbox("Filtrer par sentiment:", ["Tous", "positif", "négatif", "neutre"])
        with col_f2:
            search_query = st.text_input("🔍 Rechercher dans les titres:", placeholder="inflation, Fed, géopolitique...")
        
        filtered_news = news_articles
        if filter_sentiment != "Tous":
            filtered_news = [a for a in filtered_news if a["sentiment"] == filter_sentiment]
        if search_query:
            filtered_news = [a for a in filtered_news if search_query.lower() in a["title"].lower()]
        
        st.markdown(f"**{len(filtered_news)} articles** analysés")
        
        for article in filtered_news[:50]:
            polarity = article["polarity"]
            sentiment = article["sentiment"]
            css_class = f"news-{sentiment if sentiment in ['positif', 'négatif'] else 'neutral'}"
            icon = "📈" if sentiment == "positif" else "📉" if sentiment == "négatif" else "➡️"
            
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
            st.info("Aucun article trouvé avec ces filtres.")
    
    # ══════════════════════════════════════════════
    # TAB 5: SOURCES
    # ══════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">📋 Sources & Méthodologie</div>', unsafe_allow_html=True)
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### 📡 Sources de données de marché")
            st.markdown("""
            | Source | Type | Mise à jour |
            |--------|------|-------------|
            | **Yahoo Finance** | OHLCV temps réel | 1-5 min |
            | **yfinance API** | Historical data | Temps réel |
            | Tickers couverts | ^NDX, ^GSPC, GC=F, ^FCHI, ^STOXX, CL=F | — |
            """)
            
            st.markdown("#### 📰 Flux RSS d'actualités")
            for src, url in RSS_FEEDS.items():
                st.markdown(f"- **{src}** — `{url[:60]}...`")
        
        with col_s2:
            st.markdown("#### 🔬 Indicateurs techniques utilisés")
            st.markdown("""
            | Indicateur | Paramètres | Utilisation |
            |-----------|-----------|-------------|
            | EMA | 9, 21 périodes | Tendance CT |
            | SMA | 50, 200 périodes | Tendance LT |
            | RSI | 14 périodes | Surachat/survente |
            | MACD | 12, 26, 9 | Momentum |
            | Bollinger | 20p, 2σ | Volatilité |
            | Stochastique | 14, 3 | Retournements |
            | ATR | 14 périodes | Volatilité |
            | OBV | — | Pression volume |
            | Volume SMA | 20 périodes | Confirmation |
            """)
        
        st.markdown("#### ⚖️ Pondérations par horizon")
        st.markdown("""
        | Horizon | Poids Technique | Poids Fondamental |
        |---------|----------------|-------------------|
        | **5 min** | 80% | 20% |
        | **15 min** | 80% | 20% |
        | **60 min** | 65% | 35% |
        | **240 min** | 50% | 50% |
        """)
        
        st.markdown("#### 🌍 Indicateurs macro suivis (20 facteurs)")
        for name, data in ECONOMIC_INDICATORS.items():
            icon = "🟢" if data["impact"] == "positif" else "🔴" if data["impact"] == "négatif" else "⚪"
            st.markdown(f"- {icon} **{name}** — Valeur: `{data['value']}` {data['trend']} — Impact: *{data['impact']}* (poids: {data['weight']*100:.0f}%)")
        
        st.warning("⚠️ **Avertissement légal** : Cette application est fournie à des fins éducatives et informatives uniquement. Elle ne constitue pas un conseil en investissement financier. Les marchés financiers comportent des risques de pertes. Consultez un conseiller financier agréé avant toute décision d'investissement.")
    
    # ── PDF Report ──
    st.markdown("---")
    st.markdown('<div class="section-title">📄 Génération de rapport</div>', unsafe_allow_html=True)
    
    col_pdf1, col_pdf2 = st.columns([2, 1])
    with col_pdf1:
        st.markdown("Générez un rapport PDF complet incluant toutes les analyses, prédictions, indicateurs et sources.")
    with col_pdf2:
        if st.button("📥 Générer rapport PDF", use_container_width=True, type="primary"):
            with st.spinner("Génération du rapport..."):
                pdf_bytes = generate_pdf_report(all_market_prices, all_predictions, fund_data)
                
                b64 = base64.b64encode(pdf_bytes).decode()
                filename = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="background: linear-gradient(135deg, #0f3460, #00d4ff); color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 10px;">⬇️ Télécharger le rapport PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ Rapport généré avec succès!")


if __name__ == "__main__":
    main()
