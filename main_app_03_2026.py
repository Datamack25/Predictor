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
    if df.empty or len(df) < 20:
        return {}

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
    tf_mult = {"5 min": 0.8, "15 min": 1.2, "60 min": 2.0, "240 min": 3.5}
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
def generate_pdf_report(market_data: dict, predictions_all: dict, fund_score: dict) -> bytes:
    """Generate a PDF report — all text sanitized for latin-1."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 100, 180)
    pdf.cell(0, 15, _p("MARKET SENTIMENT PREDICTOR"), ln=True, align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 120)
    pdf.cell(0, 8, _p(f"Rapport genere le {datetime.now().strftime('%d/%m/%Y a %H:%M:%S')}"), ln=True, align="C")
    pdf.ln(5)

    pdf.set_draw_color(30, 100, 180)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    # Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 100, 180)
    pdf.cell(0, 10, _p("RESUME EXECUTIF"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)
    pdf.multi_cell(0, 6, _p(
        f"Score Fondamental: {fund_score['combined']:.1f}/100 | "
        f"Score Eco: {fund_score['eco_score']:.1f}/100 | "
        f"Score News: {fund_score['news_score']:.1f}/100\n"
        f"Actualites analysees: {fund_score['news_bull'] + fund_score['news_bear'] + fund_score['news_neutral']} | "
        f"Positives: {fund_score['news_bull']} | Negatives: {fund_score['news_bear']} | Neutres: {fund_score['news_neutral']}"
    ))
    pdf.ln(5)

    # Per market
    for market_name, data in market_data.items():
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 140, 100)
        pdf.cell(0, 10, _p(f"  {market_name}"), ln=True)

        price_info = data.get("price_info", {})
        if price_info:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 80)
            chg = price_info.get("change_pct", 0)
            try:
                price_str = f"{price_info.get('price', 0):.4g}"
                high_str  = f"{price_info.get('high', 0):.4g}"
                low_str   = f"{price_info.get('low',  0):.4g}"
            except Exception:
                price_str = high_str = low_str = "N/A"
            pdf.cell(0, 6, _p(
                f"    Prix: {price_str}  |  Variation: {chg:+.2f}%  |  H: {high_str}  L: {low_str}"
            ), ln=True)

        if market_name in predictions_all:
            preds = predictions_all[market_name]
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 180, 200)
            pdf.cell(0, 7, _p("    Predictions:"), ln=True)
            pdf.set_font("Helvetica", "", 9)
            for tf, pred in preds.items():
                direction = pred["direction"]
                bull = pred["bull_prob"]
                bear = pred["bear_prob"]
                conf = pred["confidence"]
                if direction == "HAUSSIER":
                    pdf.set_text_color(16, 140, 80)
                elif direction == "BAISSIER":
                    pdf.set_text_color(200, 50, 50)
                else:
                    pdf.set_text_color(50, 100, 200)
                pdf.cell(0, 5, _p(
                    f"      {tf:8s}: {direction:9s}  |  Hausse: {bull}%  |  Baisse: {bear}%  |  Confiance: {conf}%"
                ), ln=True)

        pdf.ln(3)

    # Economic indicators
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 100, 180)
    pdf.cell(0, 10, _p("INDICATEURS ECONOMIQUES ET MACROECONOMIQUES"), ln=True)
    pdf.ln(3)

    for name, data in ECONOMIC_INDICATORS.items():
        impact = data["impact"]
        trend  = data["trend"]
        # Convert trend arrows to ascii
        trend_ascii = ">>" if trend == "↗" else "<<" if trend == "↘" else "->"
        if impact == "positif":
            pdf.set_text_color(16, 140, 80)
        elif impact in ("négatif", "negatif"):
            pdf.set_text_color(200, 50, 50)
        else:
            pdf.set_text_color(50, 100, 200)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _p(
            f"  {name:35s}: {str(data['value']):10s} {trend_ascii}  [{impact.upper()}]  poids: {data['weight']*100:.0f}%"
        ), ln=True)

    # Sources
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 100, 180)
    pdf.cell(0, 10, _p("SOURCES ET METHODOLOGIE"), ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 140, 100)
    pdf.cell(0, 8, _p("Donnees de marche:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)
    pdf.multi_cell(0, 6, _p(
        "- Yahoo Finance (yfinance) : donnees OHLCV en temps reel et historiques\n"
        "- Intervalles: 5min, 15min, 60min, 240min\n"
        "- Tickers: ^NDX, ^GSPC, GC=F, ^FCHI, ^STOXX, CL=F"
    ))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 140, 100)
    pdf.cell(0, 8, _p("Sources d'actualites (RSS):"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)
    for source in RSS_FEEDS.keys():
        pdf.cell(0, 5, _p(f"  - {source}"), ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 140, 100)
    pdf.cell(0, 8, _p("Indicateurs techniques utilises:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)
    pdf.multi_cell(0, 6, _p(
        "- EMA 9, EMA 21, SMA 50, SMA 200\n"
        "- RSI (14 periodes)\n"
        "- MACD (12, 26, 9)\n"
        "- Bandes de Bollinger (20 periodes, 2 ecarts-types)\n"
        "- Stochastique (14, 3)\n"
        "- ATR (Average True Range)\n"
        "- OBV (On-Balance Volume)\n"
        "- Volume SMA (20 periodes)"
    ))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 140, 100)
    pdf.cell(0, 8, _p("Methodologie de prediction:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)
    pdf.multi_cell(0, 6, _p(
        "- Scoring technique: 8 indicateurs ponderes (RSI, MACD, EMA, BB, Stoch, Volume, Momentum)\n"
        "- Scoring fondamental: 20 indicateurs macro + analyse NLP des actualites (TextBlob)\n"
        "- Ponderation CT (5min/15min) = 80% tech + 20% macro\n"
        "  MT (60min) = 65% tech + 35% macro | LT (240min) = 50/50\n"
        "- Analyse de sentiment: polarite et subjectivite des titres d'articles\n"
        "AVERTISSEMENT: Ces predictions sont a des fins educatives uniquement."
    ))

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.ln(10)
    pdf.multi_cell(0, 5, _p(
        "AVERTISSEMENT LEGAL: Ce rapport est genere automatiquement a des fins informatives et educatives "
        "uniquement. Il ne constitue pas un conseil financier. Les marches financiers sont risques. "
        "Consultez un conseiller financier avant toute decision d'investissement."
    ))

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
        news_articles = fetch_news_sentiment()
        fund_data = compute_fundamental_score(news_articles)

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "📈 Graphiques OHLC",
        "🌍 Macro & Sentiment",
        "📰 Actualités",
        "🎯 Conseils Scalping",
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
    # TAB 5: CONSEILS SCALPING  ← NEW
    # ══════════════════════════════════════════════
    with tab5:
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
        scalp_pred = all_predictions[scalp_market].get(scalp_tf, {})
        scalp_df   = all_market_prices[scalp_market]["df"]
        advice     = compute_scalp_advice(scalp_df, scalp_pred, scalp_market, scalp_tf, trade_size)

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
    # TAB 6: SOURCES
    # ══════════════════════════════════════════════
    with tab6:
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
                    pdf_bytes = generate_pdf_report(all_market_prices, all_predictions, fund_data)
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
