"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          🦢  FRAGILITY DETECTION AGENT  —  Standalone Version               ║
║          Inspired by Nassim Nicholas Taleb's Framework                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL:  pip install streamlit yfinance groq scipy plotly                  ║
║            (free API via Groq — groq.com)                                    ║
║  RUN:      streamlit run fragility_agent_standalone.py                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

The agent does NOT predict Black Swans — it identifies the structural conditions
under which markets become increasingly fragile and vulnerable to extreme shocks.
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import json, re

import yfinance as yf
import urllib.request

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# § 1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CRISIS_PERIODS = {
    "2008 Financial Crisis":     ("2008-09-01", "2009-03-31",
        "Lehman collapse & global deleveraging",
        "Classic hidden fragility: compressed vol masked enormous systemic risk"),
    "2010 Flash Crash":          ("2010-05-01", "2010-06-15",
        "Algorithmic cascade, Dow -1,000 pts intraday",
        "Black Swan in the Fourth Quadrant: high uncertainty, non-linear outcome"),
    "2011 European Debt Crisis": ("2011-07-01", "2011-10-31",
        "Sovereign contagion across peripheral Europe",
        "Fragility of tightly coupled systems amplifies localised shocks"),
    "2015 China Shock":          ("2015-08-18", "2015-09-30",
        "Chinese equity crash + RMB devaluation",
        "Leverage amplifies fragility; small triggers produce outsized outcomes"),
    "2018 Vol Explosion":        ("2018-01-25", "2018-02-28",
        "VIX spike to 37, XIV inverse-VIX product collapse",
        "Compressed volatility ≠ safety; crowded short-vol trade unwind"),
    "2020 COVID Crash":          ("2020-02-19", "2020-03-23",
        "Fastest bear market in history: S&P 500 –34% in 33 days",
        "Fat-tailed distributions make rare events far more common than models predict"),
    "2022 Rate Shock":           ("2022-01-01", "2022-12-31",
        "Aggressive Fed tightening, bond-equity correlation regime change",
        "Duration risk hidden during ZIRP; regime change exposed fragility instantly"),
}

FRAGILITY_WEIGHTS = {
    "volatility_compression": 0.20,
    "vol_structure":          0.15,
    "tail_risk_demand":       0.15,
    "correlation_regime":     0.15,
    "credit_conditions":      0.20,
    "macro_regime":           0.15,
}

FRED_SERIES = {
    "BAMLH0A0HYM2": "HY Spread (bps)",
    "BAMLC0A0CM":   "IG Spread (bps)",
    "T10Y2Y":       "10Y–2Y Spread (%)",
    "T10YIE":       "10Y Breakeven Inflation",
}

MONITORING_TICKERS = ["^VIX", "^VVIX", "^GSPC", "GLD", "TLT", "HYG", "LQD"]

ASSET_CLASS_MAP = {
    "SPY":"equity","QQQ":"equity","IWM":"equity","VTI":"equity",
    "EFA":"equity","EEM":"equity","^GSPC":"equity","^NDX":"equity",
    "XLF":"equity","XLK":"equity","XLE":"equity","XLV":"equity",
    "TLT":"bond","IEF":"bond","SHY":"bond","AGG":"bond","BND":"bond",
    "HYG":"credit_hy","JNK":"credit_hy","LQD":"credit_ig",
    "GLD":"gold","IAU":"gold","SGOL":"gold",
    "UUP":"usd","SHV":"cash","BIL":"cash",
}

SCENARIO_LIBRARY = [
    dict(name="2008-Style Deleveraging",    equity=-0.40, bond=+0.15, gold=+0.05, hy=-0.30, ig=-0.08),
    dict(name="2020 COVID Liquidity Shock", equity=-0.34, bond=+0.20, gold=-0.03, hy=-0.22, ig=-0.10),
    dict(name="2022 Rate Shock (Stagflationary)", equity=-0.20, bond=-0.25, gold=+0.02, hy=-0.15, ig=-0.18),
    dict(name="Vol Explosion (Short-Vol Unwind)", equity=-0.08, bond=+0.02, gold=+0.01, hy=-0.05, ig=-0.02),
    dict(name="Geopolitical Black Swan",    equity=-0.15, bond=+0.03, gold=+0.12, hy=-0.10, ig=-0.04),
]


# ══════════════════════════════════════════════════════════════════════════════
# § 2  MARKET DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: list, period: str = "2y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        prices = prices.reindex(columns=tickers)
        return prices.dropna(how="all").ffill().bfill()
    except Exception as e:
        st.warning(f"Price fetch warning: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(lookback_days: int = 730) -> pd.DataFrame:
    """Fetch FRED macro series directly via their public CSV endpoint — no package needed."""
    out = {}
    for sid in FRED_SERIES:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
            from io import StringIO
            df = pd.read_csv(StringIO(raw), parse_dates=["DATE"], index_col="DATE")
            df.columns = [sid]
            df = df.replace(".", float("nan")).astype(float)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
            out[sid] = df[sid].loc[cutoff:]
        except Exception:
            pass
    return pd.DataFrame(out).dropna(how="all").ffill() if out else pd.DataFrame()

def market_snapshot(prices: pd.DataFrame, fred: pd.DataFrame) -> dict:
    c = {}
    if "^VIX" in prices.columns:
        v = prices["^VIX"].dropna()
        if len(v) >= 2:
            c["vix"] = round(float(v.iloc[-1]), 2)
            c["vix_pct"] = round(float((v.tail(252) <= v.iloc[-1]).mean() * 100), 1)
            c["vix_1d"] = round(float(v.iloc[-1] - v.iloc[-2]), 2)
    if "^VVIX" in prices.columns:
        vv = prices["^VVIX"].dropna()
        if len(vv) >= 1:
            c["vvix"] = round(float(vv.iloc[-1]), 2)
            if "vix" in c and c["vix"] > 0:
                c["vvix_vix"] = round(c["vvix"] / c["vix"], 3)
    if "^GSPC" in prices.columns:
        sp = prices["^GSPC"].dropna()
        if len(sp) > 22:
            c["sp500"] = round(float(sp.iloc[-1]), 2)
            c["sp500_30d"] = round(float(sp.iloc[-1] / sp.iloc[-22] - 1) * 100, 2)
    if not fred.empty:
        if "BAMLH0A0HYM2" in fred.columns:
            hy = fred["BAMLH0A0HYM2"].dropna()
            if len(hy) > 0:
                c["hy_spread"] = round(float(hy.iloc[-1]), 2)
                c["hy_pct"]    = round(float((hy.tail(252) <= hy.iloc[-1]).mean() * 100), 1)
                if len(hy) > 22:
                    c["hy_30d_chg"] = round(float(hy.iloc[-1] - hy.iloc[-22]), 2)
        if "BAMLC0A0CM" in fred.columns:
            ig = fred["BAMLC0A0CM"].dropna()
            if len(ig) > 0:
                c["ig_spread"] = round(float(ig.iloc[-1]), 2)
        if "T10Y2Y" in fred.columns:
            yc = fred["T10Y2Y"].dropna()
            if len(yc) > 0:
                c["yield_curve"]  = round(float(yc.iloc[-1]), 3)
                c["yc_inverted"]  = bool(yc.iloc[-1] < 0)
        if "T10YIE" in fred.columns:
            be = fred["T10YIE"].dropna()
            if len(be) > 0:
                c["breakeven"] = round(float(be.iloc[-1]), 2)
    c["as_of"] = prices.index[-1].strftime("%Y-%m-%d") if len(prices) else "N/A"
    return c


# ══════════════════════════════════════════════════════════════════════════════
# § 3  STATISTICAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")

def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)

def tail_risk_metrics(returns: pd.Series, confidence: float = 0.95) -> dict:
    r = returns.dropna()
    if len(r) < 30:
        return {}
    mu, sigma = float(r.mean()), float(r.std())
    skew, kurt = float(r.skew()), float(r.kurt())
    alpha = 1 - confidence
    var_h = float(np.percentile(r, alpha * 100))
    cvar_h = float(r[r <= var_h].mean()) if (r <= var_h).any() else var_h
    z = stats.norm.ppf(alpha)
    var_g = mu + sigma * z
    z_cf = z + (z**2-1)*skew/6 + (z**3-3*z)*kurt/24 - (2*z**3-5*z)*skew**2/36
    var_cf = mu + sigma * z_cf
    cumret = (1 + r).cumprod()
    max_dd = float((cumret / cumret.cummax() - 1).min())
    gap = abs((var_cf - var_g) / var_g) * 100 if var_g != 0 else 0
    return dict(
        var_hist=round(var_h, 4), cvar_hist=round(cvar_h, 4),
        var_gaussian=round(var_g, 4), var_cf=round(var_cf, 4),
        gaussian_underestimate_pct=round(float(gap), 1),
        skewness=round(skew, 4), excess_kurtosis=round(kurt, 4),
        ann_vol=round(float(sigma * np.sqrt(252)), 4),
        max_drawdown=round(max_dd, 4),
    )

def avg_pairwise_corr(returns_df: pd.DataFrame, window: int = 63) -> float:
    if returns_df.shape[1] < 2 or len(returns_df) < window:
        return 0.5
    corr = returns_df.tail(window).corr().values
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    vals = corr[mask]
    return float(np.nanmean(np.abs(vals))) if len(vals) else 0.5


# ══════════════════════════════════════════════════════════════════════════════
# § 4  FRAGILITY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class FragilityDetector:
    """Composite 0–100 fragility score from 6 weighted signals."""

    def __init__(self, prices: pd.DataFrame, fred: pd.DataFrame):
        self.prices = prices
        self.fred   = fred
        self._ret   = {}
        for col in prices.columns:
            if col not in ["^VIX","^VVIX"]:
                r = np.log(prices[col] / prices[col].shift(1)).dropna()
                if len(r) > 5:
                    self._ret[col] = r

    # ── Signal 1: Volatility Compression ──────────────────────────────────────
    def _s_vol_compression(self):
        if "^VIX" not in self.prices.columns:
            return 0.5, "VIX unavailable"
        vix = self.prices["^VIX"].dropna()
        if len(vix) < 60:
            return 0.5, "Insufficient history"
        curr = float(vix.iloc[-1])
        pct  = float((vix.tail(252) <= curr).mean())
        s    = 1.0 - pct          # low VIX → high fragility
        if "^GSPC" in self._ret:
            r = self._ret["^GSPC"]
            if len(r) > 252:
                rv_ratio = (r.tail(21).std() / max(r.tail(252).std(), 1e-9))
                s = 0.70 * s + 0.30 * max(0.0, 1.0 - rv_ratio)
        return float(np.clip(s, 0, 1)), f"VIX={curr:.1f} at {pct*100:.0f}th pct"

    # ── Signal 2: Vol Structure (VVIX/VIX) ────────────────────────────────────
    def _s_vol_structure(self):
        if "^VIX" not in self.prices.columns or "^VVIX" not in self.prices.columns:
            return 0.4, "VVIX unavailable"
        vix  = self.prices["^VIX"].dropna()
        vvix = self.prices["^VVIX"].dropna()
        idx  = vix.index.intersection(vvix.index)
        if len(idx) < 30:
            return 0.4, "Insufficient overlap"
        ratio     = (vvix.loc[idx] / vix.loc[idx])
        curr_rat  = float(ratio.iloc[-1])
        rat_pct   = float((ratio.tail(252) <= curr_rat).mean())
        vvix_pct  = float((vvix.loc[idx].tail(252) <= float(vvix.loc[idx].iloc[-1])).mean())
        s = 0.55 * rat_pct + 0.45 * vvix_pct
        return float(np.clip(s, 0, 1)), f"VVIX={float(vvix.loc[idx].iloc[-1]):.1f}, ratio={curr_rat:.2f} ({rat_pct*100:.0f}th pct)"

    # ── Signal 3: Tail Risk (Skewness + Kurtosis) ─────────────────────────────
    def _s_tail_risk(self):
        if "^GSPC" not in self._ret:
            return 0.4, "S&P 500 data unavailable"
        r = self._ret["^GSPC"].tail(63)
        if len(r) < 30:
            return 0.4, "Insufficient data"
        skew = float(r.skew())
        kurt = float(r.kurt())
        s = 0.5 * float(np.clip((-skew + 2.0) / 4.0, 0, 1)) + \
            0.5 * float(np.clip(kurt / 8.0, 0, 1))
        return float(np.clip(s, 0, 1)), f"Skew={skew:.2f}, Kurt={kurt:.2f}"

    # ── Signal 4: Correlation Regime ──────────────────────────────────────────
    def _s_correlation(self):
        assets = [t for t in ["^GSPC","GLD","TLT","HYG"] if t in self._ret]
        if len(assets) < 3:
            return 0.4, "Fewer than 3 assets"
        df = pd.DataFrame({t: self._ret[t] for t in assets}).dropna()
        if len(df) < 63:
            return 0.4, "Insufficient history"
        c63  = avg_pairwise_corr(df, 63)
        c252 = avg_pairwise_corr(df, 252)
        level_s  = float(np.clip(c63 / 0.70, 0, 1))
        change_s = float(np.clip((c63 - c252 * 0.85) / 0.20, 0, 1))
        s = 0.60 * level_s + 0.40 * max(0, change_s)
        return float(np.clip(s, 0, 1)), f"Avg |corr| 63d={c63:.2f} vs 252d={c252:.2f}"

    # ── Signal 5: Credit Conditions ───────────────────────────────────────────
    def _s_credit(self):
        if not self.fred.empty and "BAMLH0A0HYM2" in self.fred.columns:
            hy = self.fred["BAMLH0A0HYM2"].dropna()
            if len(hy) >= 30:
                curr = float(hy.iloc[-1])
                pct  = float((hy.tail(252) <= curr).mean())
                mom  = (curr - float(hy.iloc[-22])) / max(float(hy.iloc[-22]), 0.01) if len(hy) > 22 else 0
                s = 0.65 * pct + 0.35 * float(np.clip(mom * 5, 0, 1))
                return float(np.clip(s, 0, 1)), f"HY={curr:.0f}bps ({pct*100:.0f}th pct)"
        # ETF proxy
        if "HYG" in self._ret:
            hyg_r = self._ret["HYG"]
            if len(hyg_r) >= 30:
                s = float(np.clip(0.5 - hyg_r.tail(22).sum() * 5, 0, 1))
                return s, "HYG proxy (FRED unavailable)"
        return 0.4, "Credit data unavailable"

    # ── Signal 6: Macro Regime ────────────────────────────────────────────────
    def _s_macro(self):
        if self.fred.empty:
            return 0.4, "FRED data unavailable"
        scores, notes = [], []
        if "T10Y2Y" in self.fred.columns:
            yc = self.fred["T10Y2Y"].dropna()
            if len(yc) >= 2:
                curr = float(yc.iloc[-1])
                scores.append(float(np.clip((-curr + 0.3) / 1.5, 0, 1)))
                notes.append(f"Yield curve={curr:.2f}% {'⚠ INVERTED' if curr<0 else ''}")
        if "T10YIE" in self.fred.columns:
            be = self.fred["T10YIE"].dropna()
            if len(be) >= 2:
                curr = float(be.iloc[-1])
                scores.append(float(np.clip(abs(curr - 2.0) / 2.5, 0, 1)))
                notes.append(f"Breakeven={curr:.2f}%")
        if not scores:
            return 0.4, "Macro data unavailable"
        return float(np.clip(np.mean(scores), 0, 1)), " | ".join(notes)

    def score(self) -> dict:
        """Compute composite fragility score and full diagnostic dict."""
        signals = {
            "volatility_compression": self._s_vol_compression(),
            "vol_structure":          self._s_vol_structure(),
            "tail_risk_demand":       self._s_tail_risk(),
            "correlation_regime":     self._s_correlation(),
            "credit_conditions":      self._s_credit(),
            "macro_regime":           self._s_macro(),
        }
        labels = {
            "volatility_compression": "Volatility Compression",
            "vol_structure":          "Vol Structure (VVIX/VIX)",
            "tail_risk_demand":       "Tail Risk Demand",
            "correlation_regime":     "Correlation Regime",
            "credit_conditions":      "Credit Conditions",
            "macro_regime":           "Macro Regime",
        }
        total = 0.0
        breakdown = {}
        warnings  = []
        for name, (raw, note) in signals.items():
            w    = FRAGILITY_WEIGHTS[name]
            cont = raw * w * 100
            total += cont
            breakdown[name] = dict(label=labels[name], raw=raw, contribution=round(cont,2),
                                   weight=w, note=note)
            if raw > 0.65:
                warnings.append(dict(signal=labels[name], raw=raw, note=note,
                                     severity="HIGH" if raw > 0.80 else "MODERATE"))

        total = round(float(np.clip(total, 0, 100)), 1)
        if   total < 25: level, color, emoji = "CALM",     "#2ecc71", "🟢"
        elif total < 50: level, color, emoji = "VIGILANT", "#f39c12", "🟡"
        elif total < 75: level, color, emoji = "FRAGILE",  "#e67e22", "🟠"
        else:            level, color, emoji = "CRITICAL", "#e74c3c", "🔴"

        msgs = {
            "CALM":     "System appears stable. Normal risk-taking may be appropriate.",
            "VIGILANT": "Elevated signals. Monitor closely and maintain defensive hedges.",
            "FRAGILE":  "High fragility. Consider reducing risk and implementing protection.",
            "CRITICAL": "Severe systemic fragility. Immediate defensive positioning recommended.",
        }
        return dict(score=total, level=level, color=color, emoji=emoji,
                    message=msgs[level], breakdown=breakdown,
                    warnings=sorted(warnings, key=lambda x: -x["raw"]))

    def historical_score_proxy(self) -> pd.Series:
        """Fast historical fragility proxy (VIX compression + kurtosis)."""
        if "^VIX" not in self.prices.columns or len(self.prices["^VIX"].dropna()) < 252:
            return pd.Series(dtype=float)
        vix = self.prices["^VIX"].dropna()
        scores = pd.Series(index=vix.index, dtype=float)
        for i in range(252, len(vix), 5):
            pct = float((vix.iloc[max(0,i-252):i] <= float(vix.iloc[i])).mean())
            scores.iloc[i] = (1.0 - pct) * 0.60 * 100
        if "^GSPC" in self._ret:
            r = self._ret["^GSPC"]
            kurt_s = r.rolling(63).kurt().reindex(vix.index)
            scores = (scores.fillna(0) + kurt_s.clip(0,10).fillna(0) / 10 * 0.40 * 100).clip(0, 100)
        return scores.replace(0, np.nan).dropna().tail(500)


# ══════════════════════════════════════════════════════════════════════════════
# § 5  PORTFOLIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioSimulator:
    """Crisis replay, Monte Carlo (fat-tails), and shock stress tests."""

    def __init__(self, portfolio: dict, prices: pd.DataFrame = None):
        total = sum(v for v in portfolio.values() if v > 0)
        self.portfolio = {k: v/total for k, v in portfolio.items() if v > 0}
        self.tickers   = list(self.portfolio.keys())
        self._prices   = prices

    def _port_returns(self, prices: pd.DataFrame) -> pd.Series:
        avail  = [t for t in self.tickers if t in prices.columns]
        if not avail:
            return pd.Series(dtype=float)
        total_w = sum(self.portfolio[t] for t in avail)
        w = {t: self.portfolio[t]/total_w for t in avail}
        rets = prices[avail].pct_change().dropna(how="all")
        return sum(rets[t] * w[t] for t in avail)

    def crisis_replay(self, name: str) -> dict:
        if name not in CRISIS_PERIODS:
            return {"error": f"Unknown: {name}"}
        start, end, desc, note = CRISIS_PERIODS[name]
        lead = str(int(start[:4])-1) + start[4:]
        try:
            raw = yf.download(self.tickers, start=lead, end=end, auto_adjust=True, progress=False)
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            prices = prices.reindex(columns=self.tickers).dropna(how="all").ffill().bfill()
        except Exception as e:
            return {"error": str(e)}
        try:
            cp = prices.loc[start:end]
        except Exception:
            return {"error": "No data in window"}
        if len(cp) < 5:
            return {"error": "Insufficient data"}
        pr = self._port_returns(cp)
        if pr.empty:
            return {"error": "No returns computed"}
        cum  = (1 + pr).cumprod()
        ret  = float(cum.iloc[-1] - 1)
        mdd  = float((cum / cum.cummax() - 1).min())
        vol  = float(pr.std() * np.sqrt(252))
        var  = float(np.percentile(pr, 5)) if len(pr) >= 10 else np.nan
        return dict(name=name, start=start, end=end, total_return=round(ret,4),
                    max_dd=round(mdd,4), ann_vol=round(vol,4),
                    var_95=round(var,4) if not np.isnan(var) else None,
                    days=len(pr), description=desc, taleb_note=note,
                    cumulative=cum)

    def run_all_crises(self):
        rows, raw = [], {}
        for name in CRISIS_PERIODS:
            r = self.crisis_replay(name)
            raw[name] = r
            if "error" in r:
                continue
            rows.append({
                "Crisis Period":       r["name"],
                "Total Return":        f"{r['total_return']*100:.1f}%",
                "Max Drawdown":        f"{r['max_dd']*100:.1f}%",
                "Ann. Volatility":     f"{r['ann_vol']*100:.1f}%",
                "Daily VaR 95%":       f"{r['var_95']*100:.2f}%" if r["var_95"] else "N/A",
                "Duration (days)":     r["days"],
                "_ret":                r["total_return"],
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(), raw

    def monte_carlo(self, n=2000, horizon=63, vol_mult=1.0, corr_stress=1.0, fat_tails=True):
        p = self._prices
        if p is None or p.empty:
            return {"error": "No prices"}
        avail = [t for t in self.tickers if t in p.columns]
        if not avail:
            return {"error": "No matching tickers"}
        rets = p[avail].pct_change().dropna()
        if len(rets) < 30:
            return {"error": "Insufficient history"}
        mu  = rets.mean().values
        cov = rets.cov().values
        vol = np.sqrt(np.diag(cov)) * vol_mult
        corr = rets.corr().values
        if corr_stress > 1.0:
            t = min((corr_stress - 1.0), 1.0)
            corr = np.clip((1-t)*corr + t*np.ones_like(corr), -1, 1)
            np.fill_diagonal(corr, 1.0)
        cov_s = np.outer(vol, vol) * corr
        try:
            L = np.linalg.cholesky(cov_s)
        except np.linalg.LinAlgError:
            ev, evec = np.linalg.eigh(cov_s)
            cov_s = evec @ np.diag(np.maximum(ev, 1e-12)) @ evec.T
            L = np.linalg.cholesky(cov_s)
        w = np.array([self.portfolio.get(t,0) for t in avail])
        w = w / w.sum()
        np.random.seed(42)
        paths = np.zeros((n, horizon))
        for s in range(n):
            if fat_tails:
                z = stats.t.rvs(df=5, size=(horizon, len(avail)))
                z = z / np.sqrt(5/3)
            else:
                z = np.random.randn(horizon, len(avail))
            paths[s] = (z @ L.T + mu) @ w
        term = (1 + paths).prod(axis=1) - 1
        var95 = float(np.percentile(term, 5))
        cvar95 = float(term[term <= var95].mean()) if (term <= var95).any() else var95
        return dict(
            terminal=term, mean=round(float(term.mean()),4),
            median=round(float(np.median(term)),4),
            var95=round(var95,4), var99=round(float(np.percentile(term,1)),4),
            cvar95=round(cvar95,4),
            prob_loss=round(float((term<0).mean()),4),
            prob_loss10=round(float((term<-.10).mean()),4),
            prob_loss20=round(float((term<-.20).mean()),4),
            horizon=horizon, n=n, vol_mult=vol_mult, fat_tails=fat_tails,
        )

    def instant_shock(self, equity=-0.20, bond=0.05, gold=0.08, hy=-0.12, ig=-0.04, name="Shock"):
        shock_map = {"equity":equity,"bond":bond,"gold":gold,"credit_hy":hy,
                     "credit_ig":ig,"gold_miners":gold*1.8,"usd":-equity*0.15,"cash":0.0}
        asset_shocks = {}
        for t in self.tickers:
            ac = ASSET_CLASS_MAP.get(t, "equity")
            asset_shocks[t] = shock_map.get(ac, equity * 0.80)
        impact = sum(self.portfolio[t] * asset_shocks[t] for t in self.tickers)
        return dict(scenario_name=name, portfolio_impact=round(impact,4),
                    equity_shock=equity, diversification_benefit=round(impact-equity,4),
                    asset_shocks=asset_shocks)

    def current_metrics(self) -> dict:
        p = self._prices
        if p is None or p.empty:
            return {}
        pr = self._port_returns(p).dropna()
        if len(pr) < 20:
            return {}
        mu, sig = float(pr.mean()), float(pr.std())
        cum = (1+pr).cumprod()
        mdd = float((cum/cum.cummax()-1).min())
        var = float(np.percentile(pr, 5))
        cvar = float(pr[pr<=var].mean()) if (pr<=var).any() else var
        ann_ret = mu * 252
        ann_vol = sig * np.sqrt(252)
        return dict(
            ann_return=round(ann_ret,4), ann_vol=round(ann_vol,4),
            sharpe=round(ann_ret/ann_vol,3) if ann_vol>0 else 0,
            var95=round(var,4), cvar95=round(cvar,4),
            max_dd=round(mdd,4),
            curr_dd=round(float(cum.iloc[-1]/cum.max()-1),4),
            calmar=round(ann_ret/abs(mdd),3) if mdd<-0.001 else 0,
            skew=round(float(pr.skew()),4),
            kurt=round(float(pr.kurt()),4),
        )


# ══════════════════════════════════════════════════════════════════════════════
# § 6  AI AGENTS  (Claude-powered)
# ══════════════════════════════════════════════════════════════════════════════

GROQ_MODEL   = "llama-3.3-70b-versatile"

HEDGE_SYSTEM = """You are a senior quantitative risk strategist and derivatives specialist with expertise in tail-risk hedging and Nassim Taleb's antifragility framework.

Philosophy: seek CONVEXITY, not just protection. Ideal hedges have asymmetric payoffs. Align recommendations to SPECIFIC fragility signals, not generic advice. Return ONLY valid JSON."""

SCENARIO_SYSTEM = """You are a senior tail-risk strategist specializing in Taleb's framework. You identify how current fragilities could AMPLIFY a shock into a crisis. Scenarios must be mechanistically coherent: trigger → fragility amplification → contagion. Return ONLY a valid JSON array."""


def _groq_call(api_key, system, user, max_tokens=2500) -> str:
    """Single helper that calls Groq and returns the response text."""
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def run_hedging_agent(api_key, portfolio, frag_result, market, risk_metrics):
    if not GROQ_AVAILABLE:
        return {"error": "groq package not installed — run: pip install groq"}
    bd = frag_result["breakdown"]
    bd_lines = "\n".join(
        f"  {v['label']}: {v['raw']*100:.0f}/100 — {v['note']}" for v in bd.values()
    )
    warn_lines = "\n".join(
        f"  ⚠ {w['signal']} ({w['severity']}): {w['note']}"
        for w in frag_result["warnings"]
    ) or "  None"
    port_lines = "\n".join(f"  {t}: {w*100:.1f}%" for t,w in portfolio.items())
    def _f(v, s=""): return f"{v}{s}" if v is not None else "N/A"
    mkt_block = f"""
  VIX: {_f(market.get('vix'))} ({_f(market.get('vix_pct'))}th pct)
  VVIX/VIX ratio: {_f(market.get('vvix_vix'))}
  HY Spread: {_f(market.get('hy_spread'), ' bps')} ({_f(market.get('hy_pct'))}th pct)
  Yield Curve 10Y–2Y: {_f(market.get('yield_curve'))}% {'⚠ INVERTED' if market.get('yc_inverted') else ''}
  Breakeven Inflation: {_f(market.get('breakeven'), '%')}"""
    rm = risk_metrics or {}
    risk_block = f"""
  Ann. Vol: {_f(rm.get('ann_vol') and round(rm['ann_vol']*100,1), '%')}
  Sharpe: {_f(rm.get('sharpe'))}
  Daily VaR 95%: {_f(rm.get('var95') and round(rm['var95']*100,2), '%')}
  Max Drawdown: {_f(rm.get('max_dd') and round(rm['max_dd']*100,1), '%')}
  Skewness: {_f(rm.get('skew'))}  |  Excess Kurtosis: {_f(rm.get('kurt'))}"""
    prompt = f"""PORTFOLIO:\n{port_lines}\n\nFRAGILITY: {frag_result['score']}/100 ({frag_result['level']})\nSignals:\n{bd_lines}\nWarnings:\n{warn_lines}\n\nMARKET:{mkt_block}\n\nRISK METRICS:{risk_block}

Respond with JSON:
{{
  "executive_summary": "2-3 sentences on vulnerability and recommended posture",
  "overall_posture": "DEFENSIVE | BALANCED | OPPORTUNISTIC",
  "taleb_framework_note": "How current conditions map to Taleb concepts",
  "primary_recommendation": {{
    "instrument": "name",
    "action": "precise action",
    "rationale": "why most critical given specific signals",
    "sizing": "% of portfolio",
    "estimated_annual_cost": "e.g. 1.5% of notional",
    "crisis_effectiveness": "VERY HIGH | HIGH | MEDIUM | LOW",
    "convexity_profile": "how payoff scales non-linearly with stress",
    "implementation_note": "practical execution note"
  }},
  "secondary_recommendations": [
    {{"instrument":"name","action":"action","rationale":"rationale","sizing":"%","priority":"HIGH|MEDIUM|LOW","crisis_effectiveness":"VERY HIGH|HIGH|MEDIUM|LOW"}}
  ],
  "risks_to_hedging_strategy": "key risks to the hedge",
  "monitoring_triggers": ["signal 1","signal 2","signal 3"],
  "annual_cost_of_protection": "total cost estimate",
  "unhedgeable_risks": "risks to accept or size around"
}}"""
    try:
        text = _groq_call(api_key, HEDGE_SYSTEM, prompt, max_tokens=2500)
        m = re.search(r"\{[\s\S]*\}", text)
        return json.loads(m.group()) if m else {"error": "No JSON found", "_raw": text}
    except Exception as e:
        return {"error": str(e)}


def run_scenario_agent(api_key, frag_result, market, portfolio, n=5):
    if not GROQ_AVAILABLE:
        return [{"error": "groq package not installed — run: pip install groq"}]
    bd = frag_result["breakdown"]
    top = sorted([(k, v["raw"], v["note"]) for k,v in bd.items()], key=lambda x:-x[1])[:3]
    top_lines = "\n".join(f"  {i+1}. {s[0].replace('_',' ').title()} ({s[1]*100:.0f}/100): {s[2]}" for i,s in enumerate(top))
    warn_lines = "\n".join(f"  ⚠ {w['signal']}: {w['note']}" for w in frag_result["warnings"]) or "  None"
    def _f(v,s=""): return f"{v}{s}" if v is not None else "N/A"
    port_str = ", ".join(f"{t} {w*100:.0f}%" for t,w in sorted(portfolio.items(), key=lambda x:-x[1]))
    ctx = f"""Fragility: {frag_result['score']}/100 ({frag_result['level']})
Top drivers:\n{top_lines}\nWarnings:\n{warn_lines}
Market: VIX={_f(market.get('vix'))}, HY={_f(market.get('hy_spread'),' bps')}, Yield curve={_f(market.get('yield_curve'))}% {'⚠INVERTED' if market.get('yc_inverted') else ''}
Portfolio: {port_str}"""
    prompt = f"""{ctx}

Generate {n} stress scenarios. Return JSON array:
[{{
  "id": 1,
  "name": "evocative name (5-8 words)",
  "type": "FINANCIAL|MACRO|GEOPOLITICAL|TECHNICAL|LIQUIDITY|CONTAGION",
  "severity": "SEVERE|EXTREME|CATASTROPHIC",
  "probability_label": "LOW|MODERATE|ELEVATED",
  "trigger": "specific catalyst (1-2 sentences)",
  "fragility_amplification": "how detected fragility amplifies this into a crisis (2-3 sentences)",
  "taleb_annotation": "Taleb concept illustrated",
  "market_impacts": {{
    "equity_shock": -0.XX,
    "bond_move": X.XX,
    "gold_move": X.XX,
    "hy_credit_shock": -0.XX,
    "ig_credit_shock": -0.XX,
    "vix_peak_estimate": XX,
    "duration_days": XX
  }},
  "portfolio_vulnerability": "specific exposure given composition",
  "early_warning_signals": ["signal 1","signal 2","signal 3"],
  "historical_analog": "most comparable crisis and key similarity",
  "antifragile_opportunity": "how an antifragile portfolio would profit"
}}]"""
    try:
        text = _groq_call(api_key, SCENARIO_SYSTEM, prompt, max_tokens=4000)
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            return json.loads(m.group())
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            obj = json.loads(m.group())
            for v in obj.values():
                if isinstance(v, list): return v
        return [{"error":"Could not parse","_raw":text[:400]}]
    except Exception as e:
        return [{"error": str(e)}]


def run_narrative_agent(api_key, frag_result, scenarios, portfolio, market):
    if not GROQ_AVAILABLE:
        return "groq package not installed"
    sc_names = [s.get("name","") for s in scenarios[:3] if "error" not in s]
    port_str = ", ".join(f"{t} {w*100:.0f}%" for t,w in portfolio.items())
    prompt = f"""Write a 3-paragraph executive briefing for an investment committee.
Fragility Score: {frag_result['score']}/100 ({frag_result['level']})
Portfolio: {port_str}
Top scenarios: {', '.join(sc_names)}
Para 1: Current fragility assessment. Para 2: Top 2 scenario risks for this portfolio. Para 3: Recommended posture and most important action.
Style: direct, analytical, Taleb-informed. No bullets. No first person."""
    try:
        return _groq_call(api_key, SCENARIO_SYSTEM, prompt, max_tokens=700)
    except Exception as e:
        return f"Narrative generation failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# § 7  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG  = "rgba(15,23,42,0.85)"
PAPER_BG = "rgba(0,0,0,0)"
FONT_COL = "#e2e8f0"
GRID_COL = "#1e293b"

def _layout(fig, height=300, **kwargs):
    kwargs.setdefault("margin", dict(l=40, r=40, t=45, b=30))
    fig.update_layout(height=height, paper_bgcolor=PAPER_BG, plot_bgcolor=DARK_BG,
                      font={"color":FONT_COL}, **kwargs)

def chart_gauge(score, level, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix":"/100","font":{"size":34,"color":color}},
        title={"text":f"<b>Fragility Score</b><br><span style='color:{color};font-size:0.85em'>{level}</span>","font":{"size":15}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#555"},
               "bar":{"color":color,"thickness":0.25},"bgcolor":"#1e293b","borderwidth":0,
               "steps":[{"range":[0,25],"color":"rgba(46,204,113,0.15)"},
                        {"range":[25,50],"color":"rgba(243,156,18,0.15)"},
                        {"range":[50,75],"color":"rgba(230,126,34,0.15)"},
                        {"range":[75,100],"color":"rgba(231,76,60,0.15)"}],
               "threshold":{"line":{"color":color,"width":4},"thickness":0.8,"value":score}},
    ))
    fig.update_layout(height=270, paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
                      font={"color":FONT_COL}, margin=dict(l=20,r=20,t=55,b=15))
    return fig

def chart_radar(breakdown):
    labels = [v["label"] for v in breakdown.values()]
    scores = [v["raw"]*100 for v in breakdown.values()]
    labels.append(labels[0]); scores.append(scores[0])
    fig = go.Figure(go.Scatterpolar(r=scores, theta=labels, fill="toself",
        fillcolor="rgba(231,76,60,0.18)", line=dict(color="#e74c3c",width=2),
        marker=dict(size=5,color="#e74c3c")))
    fig.update_layout(polar=dict(bgcolor=PAPER_BG,
        radialaxis=dict(visible=True,range=[0,100],tickfont={"size":8},gridcolor=GRID_COL),
        angularaxis=dict(tickfont={"size":10},gridcolor=GRID_COL)),
        showlegend=False, height=320, paper_bgcolor=PAPER_BG,
        font={"color":FONT_COL}, margin=dict(l=50,r=50,t=25,b=25))
    return fig

def chart_vix(prices):
    if "^VIX" not in prices.columns:
        return go.Figure()
    vix  = prices["^VIX"].dropna().tail(504)
    vvix = prices["^VVIX"].dropna().tail(504) if "^VVIX" in prices.columns else None
    fig  = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=vix.index, y=vix.values, name="VIX",
        line=dict(color="#e74c3c",width=1.5), fill="tozeroy",
        fillcolor="rgba(231,76,60,0.08)"), secondary_y=False)
    if vvix is not None:
        fig.add_trace(go.Scatter(x=vvix.index, y=vvix.values, name="VVIX",
            line=dict(color="#6366f1",width=1.5,dash="dot")), secondary_y=True)
    fig.add_hrect(y0=0,y1=15,fillcolor="rgba(46,204,113,0.04)",line_width=0)
    fig.add_hrect(y0=30,y1=80,fillcolor="rgba(231,76,60,0.04)",line_width=0)
    fig.update_layout(title="VIX & VVIX", height=280, paper_bgcolor=PAPER_BG,
        plot_bgcolor=DARK_BG, font={"color":FONT_COL}, margin=dict(l=40,r=40,t=40,b=25),
        legend=dict(orientation="h",x=0,y=1.02),
        xaxis=dict(gridcolor=GRID_COL),
        yaxis=dict(title="VIX",gridcolor=GRID_COL),
        yaxis2=dict(title="VVIX",gridcolor=GRID_COL))
    return fig

def chart_spreads(fred):
    fig = go.Figure()
    for sid, col, name in [("BAMLH0A0HYM2","#e74c3c","HY Spread"),
                            ("BAMLC0A0CM","#3498db","IG Spread")]:
        if sid in fred.columns:
            s = fred[sid].dropna().tail(504)
            fig.add_trace(go.Scatter(x=s.index,y=s.values,name=f"{name} (bps)",
                line=dict(color=col,width=1.5)))
    _layout(fig, height=260, title="Credit Spreads (FRED)",
            xaxis=dict(gridcolor=GRID_COL), yaxis=dict(title="bps",gridcolor=GRID_COL))
    return fig

def chart_yield_curve(fred):
    if "T10Y2Y" not in fred.columns:
        return go.Figure()
    yc = fred["T10Y2Y"].dropna().tail(504)
    cols = ["#e74c3c" if v<0 else "#2ecc71" for v in yc.values]
    fig = go.Figure(go.Bar(x=yc.index,y=yc.values,marker_color=cols))
    fig.add_hline(y=0,line_dash="dash",line_color="#ffffff",opacity=0.25)
    _layout(fig, height=240, title="Yield Curve (10Y–2Y %)",
            xaxis=dict(gridcolor=GRID_COL), yaxis=dict(title="%",gridcolor=GRID_COL))
    return fig

def chart_crisis_bars(df):
    if df.empty: return go.Figure()
    vals = [float(v.replace("%","")) for v in df["Total Return"]]
    cols = ["#e74c3c" if v<-15 else "#f39c12" if v<0 else "#2ecc71" for v in vals]
    fig  = go.Figure(go.Bar(x=df["Crisis Period"],y=vals,marker_color=cols,
        text=df["Total Return"],textposition="outside"))
    _layout(fig, height=340, title="Portfolio Return During Historical Crises",
            margin=dict(l=40,r=40,t=50,b=110),
            xaxis=dict(tickangle=-30,gridcolor=GRID_COL),
            yaxis=dict(title="Return (%)",gridcolor=GRID_COL))
    return fig

def chart_mc_histogram(mc):
    if not mc or "terminal" not in mc: return go.Figure()
    term = mc["terminal"]*100
    fig  = go.Figure(go.Histogram(x=term,nbinsx=80,name="Simulated",
        marker_color="#6366f1",opacity=0.7,histnorm="probability density"))
    for val, col, label in [(mc["var95"]*100,"#f39c12","VaR 95%"),
                             (mc["cvar95"]*100,"#e74c3c","CVaR 95%")]:
        fig.add_vline(x=val,line_dash="dash",line_color=col,
            annotation_text=f"{label}: {val:.1f}%",annotation_font_color=col)
    fig.add_vline(x=0,line_color="#ffffff",opacity=0.25)
    ft = "Student-t" if mc.get("fat_tails") else "Gaussian"
    _layout(fig, height=310,
            title=f"Monte Carlo — {mc.get('horizon',63)}d horizon | {mc.get('n',2000):,} paths | {ft}",
            xaxis=dict(title="Portfolio Return (%)",gridcolor=GRID_COL),
            yaxis=dict(title="Density",gridcolor=GRID_COL))
    return fig

def chart_corr_heatmap(corr):
    if corr is None or corr.empty: return go.Figure()
    fig = px.imshow(corr.round(2),text_auto=True,color_continuous_scale="RdBu_r",zmin=-1,zmax=1,
                    title="63-Day Rolling Correlation Matrix")
    fig.update_layout(height=330,paper_bgcolor=PAPER_BG,font={"color":FONT_COL},
                      margin=dict(l=40,r=40,t=50,b=30))
    return fig

def chart_scenario_impact(results):
    if not results: return go.Figure()
    names   = [s["scenario_name"] for s in results]
    impacts = [s["portfolio_impact"]*100 for s in results]
    eq_only = [s["equity_shock"]*100 for s in results]
    cols    = ["#e74c3c" if v<-15 else "#f39c12" if v<-5 else "#2ecc71" for v in impacts]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Portfolio", x=names, y=impacts, marker_color=cols,
        text=[f"{v:.1f}%" for v in impacts], textposition="outside"))
    fig.add_trace(go.Scatter(name="Pure Equity", x=names, y=eq_only,
        mode="markers+lines", marker_color="#6366f1",
        line=dict(dash="dot"), marker_size=7))
    _layout(fig, height=340, title="Portfolio Impact Under Stress Scenarios",
            margin=dict(l=40,r=40,t=50,b=115),
            xaxis=dict(tickangle=-30,gridcolor=GRID_COL),
            yaxis=dict(title="Impact (%)",gridcolor=GRID_COL),
            legend=dict(orientation="h"), barmode="group")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# § 8  STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Fragility Agent | Taleb Framework",
                   page_icon="🦢", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
.main-hdr{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
  padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.4rem;border-left:4px solid #e74c3c}
.main-hdr h1{color:#fff;margin:0;font-size:1.9rem}
.main-hdr p{color:#a0aec0;margin:.3rem 0 0;font-size:.88rem}
.sig-card{background:#1e293b;border-radius:10px;padding:.9rem;border:1px solid #334155;margin:.25rem 0}
.sig-card h4{color:#94a3b8;margin:0 0 .35rem;font-size:.75rem;text-transform:uppercase}
.sig-card .sc{font-size:1.75rem;font-weight:700}
.sig-card p{color:#64748b;font-size:.75rem;margin:.25rem 0 0}
.warn-box{background:rgba(231,76,60,.1);border-left:3px solid #e74c3c;
  padding:.7rem 1rem;border-radius:5px;margin:.3rem 0}
.warn-box p{margin:0;color:#e74c3c;font-size:.84rem}
.rec-card{background:linear-gradient(135deg,#1e293b,#0f172a);
  border-radius:10px;padding:1.1rem;border:1px solid #334155}
.sc-card{background:#1e293b;border-radius:10px;padding:1.1rem;
  border-left:4px solid #6366f1;margin-bottom:.7rem}
[data-testid="metric-container"]{background:#1e293b;border-radius:8px;padding:.7rem}
.step-card{background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:10px;
  padding:1.1rem 1.3rem;border:1px solid #334155;margin-bottom:.5rem}
.step-card .step-num{color:#6366f1;font-size:1.4rem;font-weight:700}
.step-card p{color:#94a3b8;font-size:.87rem;line-height:1.6;margin:.3rem 0 0}
.info-banner{background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.3);
  border-radius:8px;padding:.85rem 1.1rem;margin-bottom:.8rem}
.info-banner b{color:#a5b4fc}
</style>""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "done":False,"portfolio":{"SPY":0.50,"TLT":0.25,"GLD":0.15,"HYG":0.10},
    "prices":None,"fred":None,"frag":None,"market":None,"metrics":None,
    "crisis_df":None,"crisis_raw":None,"mc":None,"sc_lib":None,"corr":None,
    "hedge":None,"scenarios":None,"narrative":None,"hist_frag":None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def fp(v,d=1):
    return f"{v*100:.{d}f}%" if v is not None and not (isinstance(v,float) and np.isnan(v)) else "N/A"
def fn(v,d=2,s=""):
    return f"{v:.{d}f}{s}" if v is not None and not (isinstance(v,float) and np.isnan(v)) else "N/A"
def sc(s): return "#2ecc71" if s<25 else "#f39c12" if s<50 else "#e67e22" if s<75 else "#e74c3c"
def badge(level):
    c={"CALM":"#2ecc71","VIGILANT":"#f39c12","FRAGILE":"#e67e22","CRITICAL":"#e74c3c"}.get(level,"#888")
    return f'<span style="background:{c};color:#fff;padding:3px 13px;border-radius:18px;font-weight:700;font-size:.88rem">{level}</span>'

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦢 Fragility Agent")
    st.caption("Taleb-Inspired Market Fragility Intelligence")
    st.divider()

    # STEP 1 — Portfolio
    st.markdown("**① Build your portfolio**")
    st.caption("Enter any Yahoo Finance tickers (e.g. SPY, TLT, GLD, AAPL) and target weights. "
               "Weights don't need to sum to 100 — they are auto-normalised.")
    port_df = st.data_editor(
        pd.DataFrame([{"Ticker":k,"Weight (%)":round(v*100,1)} for k,v in st.session_state.portfolio.items()]),
        num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "Ticker":     st.column_config.TextColumn("Ticker", width="small"),
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0, max_value=100, step=1),
        })
    valid = port_df[port_df["Ticker"].notna() & (port_df["Ticker"].str.strip()!="") & (port_df["Weight (%)"]>0)]
    if len(valid):
        tw = valid["Weight (%)"].sum()
        portfolio = {r["Ticker"].strip().upper(): r["Weight (%)"]/tw for _,r in valid.iterrows()}
    else:
        portfolio = {"SPY":0.50,"TLT":0.25,"GLD":0.15,"HYG":0.10}

    st.divider()

    # STEP 2 — API Key
    st.markdown("**② AI engine (optional)**")
    st.caption("Tabs 4 & 5 use Groq's free LLM API for AI-powered analysis. "
               "A shared key is pre-filled — or get your own free at **groq.com**.")
    _default_groq_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    anthropic_key = st.text_input("Groq API Key", type="password",
                                   value=_default_groq_key,
                                   placeholder="gsk_...",
                                   help="Used only for Tabs 4 (Hedging Intelligence) and 5 (Scenario Lab). Tabs 1–3 work without any API key. On Streamlit Cloud this is pre-filled from Secrets.")
    st.caption("Tabs 1–3 (market data, fragility score, stress tests) require no API key.")

    st.divider()

    # STEP 3 — Settings & Run
    st.markdown("**③ Configure & run**")
    st.caption("Adjust lookback period and Monte Carlo parameters, then click Run.")
    period   = st.selectbox("Data Lookback", ["1y","2y","5y"], index=1,
                             help="How far back to fetch price history. 2y is recommended for a good balance of data and relevance.")
    mc_h     = st.selectbox("Monte Carlo Horizon (days)", [21,42,63,126], index=2,
                             help="How many trading days ahead the Monte Carlo simulation looks. 63 days ≈ 3 months.")
    mc_vol   = st.slider("Vol Stress Multiplier", 1.0, 3.0, 1.0, 0.1,
                          help="Scales up the portfolio's historical volatility for stress-testing. 1.0 = no stress; 2.0 = double vol.")
    mc_corr  = st.slider("Correlation Stress Multiplier", 1.0, 2.0, 1.0, 0.1,
                          help="Pushes asset correlations toward 1 (everything crashes together). Simulates crisis correlation regimes.")

    st.divider()
    run = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True,
                    help="Fetches live data from Yahoo Finance & FRED, then runs the full fragility analysis. Takes ~30–60s.")
    if st.session_state.done:
        st.success(f"✅ Analysis ready — data as of {st.session_state.market.get('as_of','')}")
        st.caption("Re-run anytime to refresh with latest market data.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-hdr">
  <h1>🦢 Fragility Detection Agent</h1>
  <p>Inspired by Nassim Nicholas Taleb · The agent identifies fragile conditions, not predicts Black Swans ·
  Data: Yahoo Finance + FRED · AI: Groq (Llama 3.3 70B)</p>
</div>""", unsafe_allow_html=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
if run:
    prog = st.progress(0, "Fetching market data…")
    all_t = list(set(MONITORING_TICKERS + list(portfolio.keys())))
    prices  = fetch_prices(all_t, period=period)
    prog.progress(25, "Fetching FRED macro data…")
    fred    = fetch_fred()
    prog.progress(45, "Computing fragility…")
    det     = FragilityDetector(prices, fred)
    frag    = det.score()
    hist_f  = det.historical_score_proxy()
    market  = market_snapshot(prices, fred)
    # Correlation matrix
    corr_assets = [t for t in ["^GSPC","GLD","TLT","HYG","LQD"] if t in prices.columns]
    ret_df = log_returns(prices[corr_assets]) if corr_assets else pd.DataFrame()
    corr   = ret_df.tail(63).corr() if not ret_df.empty else pd.DataFrame()
    prog.progress(65, "Running portfolio simulations…")
    sim     = PortfolioSimulator(portfolio, prices)
    metrics = sim.current_metrics()
    cdf, craw = sim.run_all_crises()
    sc_lib  = [sim.instant_shock(equity=s["equity"],bond=s["bond"],gold=s["gold"],
                                  hy=s["hy"],ig=s["ig"],name=s["name"])
               for s in SCENARIO_LIBRARY]
    mc = sim.monte_carlo(n=2000,horizon=mc_h,vol_mult=mc_vol,corr_stress=mc_corr,fat_tails=True)
    prog.progress(95, "Finalising…")
    st.session_state.update(dict(
        done=True,portfolio=portfolio,prices=prices,fred=fred,
        frag=frag,hist_frag=hist_f,market=market,metrics=metrics,
        corr=corr,crisis_df=cdf,crisis_raw=craw,sc_lib=sc_lib,mc=mc,
        hedge=None,scenarios=None,narrative=None,
    ))
    prog.progress(100,"Complete!")
    st.success("✅ Analysis complete!")
    st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────
T1, T2, T3, T4, T5 = st.tabs([
    "🌡️ Market Monitor","🔍 Fragility Radar",
    "📊 Portfolio Stress Test","🛡️ Hedging Intelligence","🎬 Scenario Lab"])

PLACEHOLDER = "▶ Click **🚀 Run Full Analysis** in the sidebar to begin. Market data loads in ~30–60 seconds."

def landing_page():
    """Rich landing page shown before the user runs the analysis."""
    st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:14px;padding:2rem 2.2rem;
  border:1px solid #334155;margin-bottom:1.5rem">
  <h2 style="color:#e2e8f0;margin:0 0 .5rem">What is the Fragility Detection Agent?</h2>
  <p style="color:#94a3b8;font-size:1.01rem;line-height:1.7;margin:0">
    This tool does <b style="color:#e2e8f0">NOT</b> predict market crashes. Instead, it identifies the
    <b style="color:#f59e0b">structural conditions</b> under which financial markets become increasingly
    <em>fragile</em> — and therefore vulnerable to extreme shocks, whatever their source.
    <br><br>
    The philosophy is Nassim Nicholas Taleb's: a fragile system does not need a big cause to produce a big
    collapse. It simply needs the right <em>trigger</em>. Our job is to detect the fragility — not to forecast
    the trigger.
  </p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div style="background:#1e293b;border-radius:10px;padding:1.1rem;
          border-top:3px solid #e74c3c;height:100%">
          <h4 style="color:#e74c3c;margin:0 0 .5rem">🦢 Black Swans & Fragility</h4>
          <p style="color:#94a3b8;font-size:.88rem;line-height:1.6">
            Taleb's key insight: <em>rare, high-impact events are inevitable</em> — but only fragile systems
            are destroyed by them. Robust systems absorb shocks; <b>antifragile</b> ones benefit from them.<br><br>
            This agent measures how fragile the market is <em>right now</em> across 6 independent signals,
            combined into a single score from 0 (calm) to 100 (critical).
          </p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style="background:#1e293b;border-radius:10px;padding:1.1rem;
          border-top:3px solid #6366f1;height:100%">
          <h4 style="color:#6366f1;margin:0 0 .5rem">📡 What it measures</h4>
          <p style="color:#94a3b8;font-size:.88rem;line-height:1.6">
            <b style="color:#e2e8f0">• Volatility compression</b> — Low VIX signals complacency, not safety.<br>
            <b style="color:#e2e8f0">• Vol-of-vol structure</b> — VVIX/VIX ratio detects hidden tail demand.<br>
            <b style="color:#e2e8f0">• Tail risk demand</b> — Are smart players quietly buying protection?<br>
            <b style="color:#e2e8f0">• Correlation regime</b> — When assets move together, diversification fails.<br>
            <b style="color:#e2e8f0">• Credit conditions</b> — HY spreads are the market's stress barometer.<br>
            <b style="color:#e2e8f0">• Macro regime</b> — Yield curve and inflation signal systemic risk.
          </p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div style="background:#1e293b;border-radius:10px;padding:1.1rem;
          border-top:3px solid #2ecc71;height:100%">
          <h4 style="color:#2ecc71;margin:0 0 .5rem">📊 What you get</h4>
          <p style="color:#94a3b8;font-size:.88rem;line-height:1.6">
            <b style="color:#e2e8f0">Tab 1 — Market Monitor:</b> Live VIX, spreads, yield curve, correlation heatmap.<br><br>
            <b style="color:#e2e8f0">Tab 2 — Fragility Radar:</b> Composite score + 6-signal breakdown.<br><br>
            <b style="color:#e2e8f0">Tab 3 — Stress Test:</b> Crisis replays, Monte Carlo with fat tails, scenario shocks.<br><br>
            <b style="color:#e2e8f0">Tab 4 — Hedging AI:</b> AI-powered hedging strategy recommendations.<br><br>
            <b style="color:#e2e8f0">Tab 5 — Scenario Lab:</b> AI-generated forward-looking stress scenarios.
          </p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""<div style="background:rgba(99,102,241,.08);border:1px solid #6366f1;
      border-radius:10px;padding:1.1rem 1.4rem;margin-bottom:1.2rem">
      <h4 style="color:#a5b4fc;margin:0 0 .6rem">🚀 How to use this tool (3 steps)</h4>
      <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
        <div style="flex:1;min-width:160px">
          <div style="color:#6366f1;font-weight:700;font-size:1.1rem">① Build your portfolio</div>
          <div style="color:#94a3b8;font-size:.87rem;margin-top:.3rem">
            In the sidebar, enter your portfolio tickers (e.g. SPY, TLT, GLD) and their weights.
            Any Yahoo Finance ticker works.
          </div>
        </div>
        <div style="flex:1;min-width:160px">
          <div style="color:#6366f1;font-weight:700;font-size:1.1rem">② Click Run Analysis</div>
          <div style="color:#94a3b8;font-size:.87rem;margin-top:.3rem">
            The agent fetches live data from Yahoo Finance &amp; FRED (the US Federal Reserve database),
            then computes all metrics. No paid API key needed for the first 3 tabs.
          </div>
        </div>
        <div style="flex:1;min-width:160px">
          <div style="color:#6366f1;font-weight:700;font-size:1.1rem">③ Explore the tabs</div>
          <div style="color:#94a3b8;font-size:.87rem;margin-top:.3rem">
            Navigate through the 5 tabs. For AI-powered hedging and scenario analysis (Tabs 4 &amp; 5),
            a Groq API key is pre-filled — or get your own free at groq.com.
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:rgba(245,158,11,.06);border-left:4px solid #f59e0b;
      border-radius:6px;padding:.8rem 1.1rem">
      <span style="color:#f59e0b;font-weight:700">⚠ Disclaimer</span>
      <span style="color:#94a3b8;font-size:.87rem"> — This tool is an academic research project built for
      a Master's thesis at EADA Business School. It is NOT financial advice. All outputs are for
      educational and analytical purposes only. Past crisis patterns do not guarantee future behaviour.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Get started:** Build your portfolio in the sidebar, then click **🚀 Run Full Analysis**.")

# ══════ TAB 1 — MARKET MONITOR ════════════════════════════════════════════════
with T1:
    if not st.session_state.done:
        landing_page()
    else:
        st.markdown("""<div style="background:rgba(99,102,241,.07);border-left:4px solid #6366f1;
          border-radius:6px;padding:.75rem 1.1rem;margin-bottom:1rem">
          <b style="color:#a5b4fc">🌡️ Market Monitor</b>
          <span style="color:#94a3b8;font-size:.9rem"> — Real-time snapshot of the key market stress indicators.
          <b style="color:#e2e8f0">VIX</b> measures implied volatility (the market's "fear gauge"). Low VIX is NOT safety —
          it often signals dangerous complacency. <b style="color:#e2e8f0">HY spreads</b> (High-Yield credit spreads)
          measure how much extra yield investors demand to hold risky bonds — rising spreads signal credit stress.
          The <b style="color:#e2e8f0">yield curve</b> (10Y minus 2Y rate) has historically inverted before every US recession.
          </span></div>""", unsafe_allow_html=True)
        m = st.session_state.market
        # Always-available metrics (Yahoo Finance)
        c1, c2, c3 = st.columns(3)
        c1.metric("VIX", fn(m.get("vix")), f"{m.get('vix_1d',0):+.2f} (1d)" if m.get("vix_1d") else None,
                  help="CBOE Volatility Index — measures 30-day implied vol of S&P 500 options. <15 = complacent; >30 = stressed; >40 = crisis.")
        c1.caption(f"{m.get('vix_pct','?')}th percentile vs last 12 months")
        c2.metric("VVIX", fn(m.get("vvix")),
                  help="Volatility of VIX itself. A high VVIX/VIX ratio means tail-risk demand is elevated relative to headline fear — a subtle fragility signal.")
        c2.caption(f"VVIX/VIX ratio: {fn(m.get('vvix_vix'))}")
        c3.metric("S&P 500 30d", f"{fn(m.get('sp500_30d'),1)}%",
                  help="S&P 500 total return over the last 30 trading days. Strong recent returns can mask building fragility — a classic Talebian trap.")

        # FRED metrics — shown only when data loaded successfully
        fred_cols = []
        if m.get("hy_spread") is not None:
            fred_cols.append("hy")
        if m.get("yield_curve") is not None:
            fred_cols.append("yc")
        if fred_cols:
            fx = st.columns(len(fred_cols))
            idx = 0
            if "hy" in fred_cols:
                fx[idx].metric("HY Spread", f"{fn(m.get('hy_spread'))} bps",
                               f"{m.get('hy_30d_chg',0):+.0f} bps (30d)" if m.get("hy_30d_chg") else None,
                               help="High-Yield credit spread (basis points above Treasuries). Rising spreads = credit stress. >600bps historically signals serious distress.")
                fx[idx].caption(f"{m.get('hy_pct','?')}th percentile vs last 12 months")
                idx += 1
            if "yc" in fred_cols:
                fx[idx].metric("Yield Curve (10Y–2Y)", f"{fn(m.get('yield_curve'))}%",
                               "⚠ INVERTED" if m.get("yc_inverted") else "Normal",
                               help="10-year minus 2-year Treasury yield. Negative (inverted) has preceded every US recession since 1970 with a 6–18 month lag.")
                idx += 1

        st.divider()
        cl,cr = st.columns([3,2])
        with cl:
            st.plotly_chart(chart_vix(st.session_state.prices), use_container_width=True)
        with cr:
            if st.session_state.fred is not None and not st.session_state.fred.empty:
                st.plotly_chart(chart_yield_curve(st.session_state.fred), use_container_width=True)

        if st.session_state.fred is not None and not st.session_state.fred.empty:
            st.plotly_chart(chart_spreads(st.session_state.fred), use_container_width=True)

        if st.session_state.corr is not None and not st.session_state.corr.empty:
            st.subheader("Cross-Asset Correlation (63-day)")
            st.plotly_chart(chart_corr_heatmap(st.session_state.corr), use_container_width=True)

# ══════ TAB 2 — FRAGILITY RADAR ═══════════════════════════════════════════════
with T2:
    if not st.session_state.done:
        st.info(PLACEHOLDER)
    else:
        st.markdown("""<div style="background:rgba(231,76,60,.07);border-left:4px solid #e74c3c;
          border-radius:6px;padding:.75rem 1.1rem;margin-bottom:.8rem">
          <b style="color:#f87171">🔍 Fragility Radar</b>
          <span style="color:#94a3b8;font-size:.9rem"> — The composite fragility score aggregates 6 independent signals
          into a single number from 0 to 100. It does <em>not</em> tell you when a crash will happen —
          it tells you how vulnerable the system is <em>right now</em> to an adverse shock,
          regardless of its source. Think of it as measuring how much dry wood is in the forest,
          not predicting which spark will start the fire.
          </span></div>""", unsafe_allow_html=True)

        with st.expander("📖 How is the score calculated? — Signal breakdown guide"):
            st.markdown("""
**The composite score is a weighted sum of 6 independently-measured fragility signals:**

| Signal | Weight | What it captures |
|---|---|---|
| **Volatility Compression** | 20% | Low, compressed VIX relative to its recent history. Complacency is fragility in disguise. |
| **Credit Conditions** | 20% | High-yield spread level and trend. Credit markets are the first to sense systemic stress. |
| **Vol Structure (VVIX/VIX)** | 15% | The ratio of VIX volatility to VIX itself. A high ratio signals that sophisticated options traders are quietly paying up for tail hedges. |
| **Tail Risk Demand** | 15% | The excess kurtosis of recent S&P 500 returns. Fat tails in recent data imply the distribution is non-Gaussian — Taleb's Fourth Quadrant. |
| **Correlation Regime** | 15% | Cross-asset correlations. In crises, everything correlates toward 1. Rising correlations = diversification is failing. |
| **Macro Regime** | 15% | Yield curve inversion, inflation breakevens. Macro imbalances are long-fuse fragility. |

**Score interpretation:**
- 🟢 **0–25 CALM** — System appears stable. Low fragility signals across the board.
- 🟡 **25–50 VIGILANT** — Elevated signals in one or more areas. Monitor closely.
- 🟠 **50–75 FRAGILE** — Multiple signals elevated. The system is meaningfully fragile. Consider defensive positioning.
- 🔴 **75–100 CRITICAL** — Severe multi-signal fragility. The system is in a highly vulnerable state.
""")
        fr = st.session_state.frag
        cl, cr = st.columns([1,1])
        with cl:
            st.plotly_chart(chart_gauge(fr["score"],fr["level"],fr["color"]), use_container_width=True)
            st.markdown(f'<div style="text-align:center;margin-top:-8px">{badge(fr["level"])}</div>',
                        unsafe_allow_html=True)
        with cr:
            st.markdown("#### Interpretation")
            st.markdown(f'<div style="background:rgba(0,0,0,.2);border-left:4px solid {fr["color"]};'
                        f'padding:.9rem;border-radius:6px">{fr["message"]}</div>', unsafe_allow_html=True)
            st.markdown("---")
            warns = fr.get("warnings",[])
            if warns:
                st.markdown(f"**⚠ Active Warnings ({len(warns)})**")
                for w in warns:
                    sc2 = "#e74c3c" if w["severity"]=="HIGH" else "#f39c12"
                    st.markdown(f'<div class="warn-box"><p><b style="color:{sc2}">[{w["severity"]}]</b> '
                                f'{w["signal"]}<br><span style="color:#a0aec0">{w["note"]}</span></p></div>',
                                unsafe_allow_html=True)
            else:
                st.success("✓ No high-severity warnings active.")

        st.divider()
        st.subheader("Signal Breakdown")
        bd = fr["breakdown"]
        cols = st.columns(3)
        for i,(name,data) in enumerate(bd.items()):
            raw = data["raw"]
            cc  = "#2ecc71" if raw<0.33 else "#f39c12" if raw<0.66 else "#e74c3c"
            with cols[i%3]:
                st.markdown(f'<div class="sig-card"><h4>{data["label"]}</h4>'
                            f'<div class="sc" style="color:{cc}">{raw*100:.0f}/100</div>'
                            f'<p>Weight {data["weight"]*100:.0f}% → {data["contribution"]:.1f}pts</p>'
                            f'<p style="color:#94a3b8;margin-top:.3rem">{data["note"]}</p></div>',
                            unsafe_allow_html=True)

        st.divider()
        ra,rb = st.columns([1,1])
        with ra:
            st.subheader("Fragility Spider")
            st.plotly_chart(chart_radar(bd), use_container_width=True)
        with rb:
            st.subheader("Weighted Contributions")
            cdf2 = pd.DataFrame({"Signal":[v["label"] for v in bd.values()],
                                  "Pts":[v["contribution"] for v in bd.values()]}).sort_values("Pts")
            fig2 = px.bar(cdf2,x="Pts",y="Signal",orientation="h",
                          color="Pts",color_continuous_scale=["#2ecc71","#f39c12","#e74c3c"],
                          range_color=[0,20],text="Pts")
            fig2.update_traces(texttemplate="%{text:.1f}pts",textposition="outside")
            fig2.update_layout(height=290,paper_bgcolor=PAPER_BG,plot_bgcolor=DARK_BG,
                               font={"color":FONT_COL},margin=dict(l=10,r=40,t=10,b=10),
                               coloraxis_showscale=False,showlegend=False,
                               xaxis=dict(title="Points",gridcolor=GRID_COL),yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)

        hf = st.session_state.hist_frag
        if hf is not None and len(hf)>10:
            st.subheader("Historical Fragility Proxy")
            st.caption("Simplified proxy (VIX compression + kurtosis) — indicative, not exact composite score.")
            fig_hf = go.Figure(go.Scatter(x=hf.index,y=hf.values,fill="tozeroy",
                line=dict(color="#6366f1",width=1.5),fillcolor="rgba(99,102,241,.12)"))
            fig_hf.add_hline(y=50,line_dash="dash",line_color="#f39c12",opacity=0.5)
            fig_hf.add_hline(y=75,line_dash="dash",line_color="#e74c3c",opacity=0.5)
            fig_hf.update_layout(height=200,paper_bgcolor=PAPER_BG,plot_bgcolor=DARK_BG,
                                  font={"color":FONT_COL},margin=dict(l=40,r=40,t=10,b=25),
                                  yaxis=dict(range=[0,100],gridcolor=GRID_COL),
                                  xaxis=dict(gridcolor=GRID_COL))
            st.plotly_chart(fig_hf, use_container_width=True)

# ══════ TAB 3 — PORTFOLIO STRESS TEST ═════════════════════════════════════════
with T3:
    if not st.session_state.done:
        st.info(PLACEHOLDER)
    else:
        st.markdown("""<div style="background:rgba(52,152,219,.07);border-left:4px solid #3498db;
          border-radius:6px;padding:.75rem 1.1rem;margin-bottom:.8rem">
          <b style="color:#60a5fa">📊 Portfolio Stress Test</b>
          <span style="color:#94a3b8;font-size:.9rem"> — Three complementary ways to understand your portfolio's
          vulnerability: (1) <b style="color:#e2e8f0">Historical crisis replay</b> — what would your specific portfolio
          have returned during past crises? (2) <b style="color:#e2e8f0">Monte Carlo simulation</b> with fat tails —
          what does the distribution of possible outcomes look like over the next 63 days?
          (3) <b style="color:#e2e8f0">Scenario shock analysis</b> — instant impact of predefined market shocks.
          </span></div>""", unsafe_allow_html=True)
        rm   = st.session_state.metrics
        mc   = st.session_state.mc
        cdf  = st.session_state.crisis_df
        slib = st.session_state.sc_lib
        port = st.session_state.portfolio

        cp, cm = st.columns([1,2])
        with cp:
            pie = px.pie(names=list(port.keys()),values=[v*100 for v in port.values()],
                         color_discrete_sequence=px.colors.qualitative.Set3)
            pie.update_traces(textposition="inside",textinfo="percent+label")
            pie.update_layout(title="Portfolio Allocation",height=260,
                              paper_bgcolor=PAPER_BG,font={"color":FONT_COL},
                              margin=dict(l=5,r=5,t=40,b=5))
            st.plotly_chart(pie, use_container_width=True)
        with cm:
            st.subheader("Current Risk Metrics")
            if rm:
                a,b,c = st.columns(3)
                a.metric("Ann. Return",  fp(rm.get("ann_return")),
                         help="Annualised portfolio return based on historical price data.")
                b.metric("Ann. Vol",     fp(rm.get("ann_vol")),
                         help="Annualised volatility (standard deviation of daily returns × √252). Higher vol = wider range of outcomes.")
                c.metric("Sharpe Ratio", fn(rm.get("sharpe")),
                         help="Return per unit of risk (Ann. Return ÷ Ann. Vol). Sharpe >1 is considered good; <0 means you're losing money for the risk taken.")
                d,e,f = st.columns(3)
                d.metric("VaR 95% (daily)", fp(rm.get("var95")),
                         help="Value-at-Risk: the worst expected daily loss 95% of the time. On 1 in 20 trading days, losses will EXCEED this number. Based on Cornish-Fisher adjustment for fat tails.")
                e.metric("CVaR 95% (daily)",fp(rm.get("cvar95")),
                         help="Conditional VaR (Expected Shortfall): the AVERAGE loss on the worst 5% of days. This is what Taleb prefers over VaR — it captures the severity of tail events, not just their threshold.")
                f.metric("Max Drawdown",    fp(rm.get("max_dd")),
                         help="The largest peak-to-trough decline in the portfolio's history. A key measure of downside risk that VaR can miss.")
                g,h,ii = st.columns(3)
                g.metric("Calmar Ratio", fn(rm.get("calmar")),
                         help="Ann. Return ÷ Max Drawdown. Measures return relative to worst historical loss. Higher is better.")
                h.metric("Skewness",     fn(rm.get("skew")),
                         help="Negative skewness means the return distribution has a longer left tail — large losses are more likely than large gains. Most equity portfolios are negatively skewed.")
                ii.metric("Excess Kurtosis", fn(rm.get("kurt")),
                          help="How 'fat' the tails of the return distribution are vs a normal distribution. Kurtosis >0 means extreme events occur more often than Gaussian models predict. Taleb's core critique of standard finance.")
                # Cornish-Fisher explanation
                kurt_v = rm.get("kurt",0) or 0
                if kurt_v > 1.5:
                    st.warning(f"⚠ Excess kurtosis = {kurt_v:.2f} → fat tails detected. "
                               f"Standard Gaussian VaR systematically underestimates your true tail risk. "
                               f"The VaR/CVaR figures above use the Cornish-Fisher adjustment to correct for this.")
                with st.expander("📖 What is Cornish-Fisher VaR? Why does it matter?"):
                    st.markdown("""
**The problem with Gaussian (normal distribution) VaR:**
Standard risk models assume asset returns follow a bell curve. But real market returns have **fat tails** —
extreme moves happen far more often than the normal distribution predicts.
In 2008, Goldman Sachs' CFO famously said they were seeing "25-standard-deviation events, several days in a row."
Under a normal distribution, a 25-sigma event should happen once in the lifetime of the universe.

**The Cornish-Fisher correction:**
This adjustment modifies the VaR calculation to account for the actual **skewness** and **kurtosis** of your returns.
If your portfolio has excess kurtosis (fat tails) and negative skewness (longer left tail),
Cornish-Fisher VaR will be *larger* than Gaussian VaR — correctly warning you that your losses could be worse.

**This is Taleb's central critique of modern finance:** models that assume Gaussian distributions systematically
underestimate tail risk, creating a false sense of security — exactly the fragility we are trying to measure.
""")
            else:
                st.warning("Risk metrics unavailable.")

        st.divider()
        st.subheader("Historical Crisis Replay")
        st.caption("How would your specific portfolio have performed during each major historical market crisis? "
                   "Returns are calculated using your actual ticker weights applied to historical price data. "
                   "Note: tickers that didn't exist during a crisis period are excluded from that period's calculation.")
        if cdf is not None and not cdf.empty:
            disp = [c for c in cdf.columns if not c.startswith("_")]
            st.dataframe(cdf[disp], use_container_width=True, hide_index=True)
            st.plotly_chart(chart_crisis_bars(cdf), use_container_width=True)
        else:
            st.info("Crisis data unavailable for this portfolio.")

        st.divider()
        st.subheader(f"Monte Carlo Simulation — {mc.get('horizon',63)}-Day Horizon")
        with st.expander("📖 What is Monte Carlo with fat tails? Why not just use a normal distribution?"):
            st.markdown("""
**Standard Monte Carlo** simulates thousands of random price paths assuming returns follow a normal (Gaussian) distribution.
The problem: real markets crash far more severely and more often than Gaussian models predict.

**This simulation uses a Student-t distribution with df=5** (heavy fat tails) to generate each random shock.
Student-t with 5 degrees of freedom produces extreme outliers roughly 7× more often than a normal distribution.
This is a more honest reflection of how financial markets actually behave.

**How to read the histogram:**
- The x-axis shows portfolio returns over the simulation horizon
- The distribution's left tail represents the range of bad outcomes
- **VaR 95%** (orange line): You should expect to lose more than this amount only 5% of the time
- **CVaR 95%** (red line): The *average* loss in the worst 5% of scenarios — what you'd expect to lose in a true bad outcome

**Vol and Correlation Stress Multipliers** (set in the sidebar) let you simulate crisis conditions:
- Vol ×2 = twice as volatile as the recent historical average
- Corr ×1.5 = assets start moving together (diversification partially failing)

The wider and heavier the left tail, the more fragile your portfolio is to tail events.
""")
        if mc and "terminal" in mc:
            ma,mb,mc2,md,me = st.columns(5)
            ma.metric("Median",      fp(mc.get("median")))
            mb.metric("VaR 95%",     fp(mc.get("var95")))
            mc2.metric("CVaR 95%",   fp(mc.get("cvar95")))
            md.metric("P(Loss>10%)", fp(mc.get("prob_loss10")))
            me.metric("P(Loss>20%)", fp(mc.get("prob_loss20")))
            st.plotly_chart(chart_mc_histogram(mc), use_container_width=True)
            if mc.get("vol_mult",1)>1 or mc.get("fat_tails"):
                info_parts = []
                if mc.get("fat_tails"): info_parts.append("Student-t fat tails (df=5)")
                if mc.get("vol_mult",1)>1: info_parts.append(f"Vol ×{mc['vol_mult']:.1f}")
                st.info("🔧 " + " | ".join(info_parts))
        else:
            st.warning("Monte Carlo failed.")

        st.divider()
        st.subheader("Scenario Shock Library")
        if slib:
            st.plotly_chart(chart_scenario_impact(slib), use_container_width=True)
            sdf = pd.DataFrame([{"Scenario":s["scenario_name"],
                                  "Portfolio Impact":fp(s["portfolio_impact"]),
                                  "Equity Shock":fp(s["equity_shock"]),
                                  "Diversif. Benefit":fp(s["diversification_benefit"])}
                                 for s in slib])
            st.dataframe(sdf, use_container_width=True, hide_index=True)

# ══════ TAB 4 — HEDGING INTELLIGENCE ══════════════════════════════════════════
with T4:
    if not st.session_state.done:
        st.info(PLACEHOLDER)
    else:
        st.markdown("""<div style="background:rgba(46,204,113,.07);border-left:4px solid #2ecc71;
          border-radius:6px;padding:.75rem 1.1rem;margin-bottom:.8rem">
          <b style="color:#4ade80">🛡️ Hedging Intelligence</b>
          <span style="color:#94a3b8;font-size:.9rem"> — The AI analyses your specific portfolio composition,
          the current fragility breakdown, and live market conditions to recommend calibrated hedging strategies.
          The philosophy is Talebian: seek <b style="color:#e2e8f0">convexity</b> — hedges that cost little in
          normal times but pay off explosively in crises. This is NOT standard portfolio insurance; it is
          structured tail-risk protection.
          </span></div>""", unsafe_allow_html=True)

        with st.expander("📖 What is convexity? Why does Taleb prefer options over linear hedges?"):
            st.markdown("""
**Convexity** means a non-linear payoff profile. An investment is convex if it gains more when things go right
than it loses when things go wrong — or in the hedging context, if its protective value *accelerates* as markets
deteriorate.

**Linear hedges** (e.g. shorting SPY to hedge equity risk) are symmetric: you lose on the hedge when markets rise,
and gain when they fall. They are expensive to maintain and cap your upside.

**Convex hedges** (e.g. out-of-the-money put options) cost a small, known premium. If nothing happens, you lose
only the premium. But if a tail event occurs, the payoff can be 10x–50x the cost. This is the "barbell strategy"
Taleb advocates: be safe in most scenarios, have explosive upside in extreme ones.

**Key metrics in this tab:**
- **Sizing** — what % of portfolio notional to allocate to the hedge
- **Annual Cost** — the ongoing drag from maintaining the hedge (option premiums, etc.)
- **Crisis Effectiveness** — how well the hedge performs specifically during tail events
- **Convexity Profile** — the non-linear payoff story: why this hedge accelerates in value during a crisis
""")

        st.subheader("🛡️ AI-Powered Hedging Recommendations")
        st.caption("The AI analyses the fragility breakdown, portfolio exposures, and market conditions "
                   "to recommend calibrated hedging strategies framed around Taleb's convexity principle.")
        if not anthropic_key:
            st.warning("Enter your **Groq API key** in the sidebar (Step ②) to unlock this tab.")
        elif not GROQ_AVAILABLE:
            st.error("`groq` package not installed. Run: `pip install groq`")
        else:
            cb,ci = st.columns([1,3])
            with cb:
                run_h = st.button("🤖 Generate Hedging Analysis", type="primary")
            with ci:
                fr2 = st.session_state.frag
                st.markdown(f'Fragility: <b style="color:{fr2["color"]}">{fr2["score"]:.0f}/100 — {fr2["level"]}</b>',
                            unsafe_allow_html=True)
            if run_h:
                with st.spinner("Claude is reasoning about hedging strategies…"):
                    st.session_state.hedge = run_hedging_agent(
                        anthropic_key, st.session_state.portfolio,
                        st.session_state.frag, st.session_state.market,
                        st.session_state.metrics)

            hr = st.session_state.hedge
            if hr:
                if "error" in hr:
                    st.error(f"Error: {hr['error']}")
                else:
                    pc_map = {"DEFENSIVE":"#e74c3c","BALANCED":"#f39c12","OPPORTUNISTIC":"#2ecc71"}
                    posture = hr.get("overall_posture","BALANCED")
                    pc = pc_map.get(posture,"#888")
                    st.markdown(f'<div style="background:rgba(0,0,0,.25);border-left:4px solid {pc};'
                                f'padding:1rem 1.2rem;border-radius:8px;margin-bottom:1rem">'
                                f'<b style="color:{pc}">Posture: {posture}</b><br><br>'
                                f'{hr.get("executive_summary","")}</div>', unsafe_allow_html=True)
                    if hr.get("taleb_framework_note"):
                        st.markdown(f'<div style="background:rgba(99,102,241,.1);border-left:3px solid #6366f1;'
                                    f'padding:.7rem 1rem;border-radius:5px;font-size:.86rem;margin-bottom:1rem">'
                                    f'📖 <b>Taleb Framework</b> — {hr["taleb_framework_note"]}</div>',
                                    unsafe_allow_html=True)
                    st.divider()
                    pr = hr.get("primary_recommendation",{})
                    if pr:
                        st.markdown("#### 🥇 Primary Recommendation")
                        pa,pb = st.columns([2,1])
                        with pa:
                            st.markdown(f'<div class="rec-card">'
                                        f'<h3 style="color:#2ecc71;margin:0">{pr.get("instrument","")}</h3>'
                                        f'<p style="color:#94a3b8;margin:.2rem 0 .7rem">Action: '
                                        f'<b style="color:#e2e8f0">{pr.get("action","")}</b></p>'
                                        f'<p style="font-size:.9rem">{pr.get("rationale","")}</p>'
                                        f'<p style="color:#94a3b8;font-size:.83rem;margin-top:.5rem">'
                                        f'<b>Impl:</b> {pr.get("implementation_note","")}</p>'
                                        f'<p style="color:#6366f1;font-size:.83rem">'
                                        f'<b>Convexity:</b> {pr.get("convexity_profile","")}</p></div>',
                                        unsafe_allow_html=True)
                        with pb:
                            st.metric("Sizing", pr.get("sizing","N/A"))
                            st.metric("Annual Cost", pr.get("estimated_annual_cost","N/A"))
                            eff = pr.get("crisis_effectiveness","N/A")
                            ec = "#2ecc71" if "HIGH" in eff else "#f39c12"
                            st.markdown(f'<div style="text-align:center;margin-top:.5rem">'
                                        f'<span style="color:{ec};font-weight:700">Effectiveness: {eff}</span></div>',
                                        unsafe_allow_html=True)
                    sec = hr.get("secondary_recommendations",[])
                    if sec:
                        st.markdown("#### 🥈 Secondary Recommendations")
                        for s in sec:
                            with st.expander(f"{s.get('instrument','')} — {s.get('priority','')} priority | {s.get('crisis_effectiveness','')}"):
                                st.markdown(f"**Action:** {s.get('action','')}")
                                st.markdown(f"**Rationale:** {s.get('rationale','')}")
                                st.markdown(f"**Sizing:** {s.get('sizing','')}")
                    st.divider()
                    trigs = hr.get("monitoring_triggers",[])
                    if trigs:
                        st.markdown("#### 📡 Monitoring Triggers")
                        for t in trigs:
                            st.markdown(f"• {t}")
                    if hr.get("risks_to_hedging_strategy"):
                        st.warning(f"⚠ **Risks to strategy:** {hr['risks_to_hedging_strategy']}")
                    if hr.get("annual_cost_of_protection"):
                        st.info(f"💰 **Total annual cost of protection:** {hr['annual_cost_of_protection']}")

# ══════ TAB 5 — SCENARIO LAB ══════════════════════════════════════════════════
with T5:
    if not st.session_state.done:
        st.info(PLACEHOLDER)
    else:
        st.markdown("""<div style="background:rgba(155,89,182,.08);border-left:4px solid #9b59b6;
          border-radius:6px;padding:.75rem 1.1rem;margin-bottom:.8rem">
          <b style="color:#c084fc">🎬 Scenario Lab</b>
          <span style="color:#94a3b8;font-size:.9rem"> — Standard stress-testing replays the past.
          This tab generates <b style="color:#e2e8f0">forward-looking hypothetical scenarios</b> grounded in the
          <em>current</em> fragility signals — scenarios that have not happened yet, but are mechanistically plausible
          given the specific vulnerabilities detected today. This is NOT forecasting. It is structured imagination
          about the tails, guided by Taleb's framework.
          </span></div>""", unsafe_allow_html=True)

        with st.expander("📖 How does forward-looking scenario generation work? Why is it different from historical stress-testing?"):
            st.markdown("""
**Historical stress-testing** asks: "What would my portfolio have returned during the 2008 crisis?"
It is useful but limited — the next crisis will not look exactly like the last one.

**Forward-looking scenario generation** asks: "Given the specific fragility signals active *right now*, what kind of shock would be most damaging — and through what mechanism?"

The AI is given your current fragility profile (the 6 signals and their levels), your portfolio composition, and the current market snapshot. It then reasons about:
1. **What conditions make the system fragile right now?** (e.g. compressed vol + high correlation)
2. **What kind of shock would those conditions amplify most violently?** (e.g. a liquidity event would be fatal under compressed vol)
3. **Through what mechanism would contagion spread?** (e.g. forced deleveraging by risk-parity funds)

Each scenario includes:
- **Trigger** — the specific initial catalyst
- **Fragility amplification** — how the detected fragility makes the trigger worse
- **Taleb annotation** — which concept from Antifragile/Black Swan applies
- **Market impacts** — quantified estimates for stress-testing your portfolio
- **Early warning signals** — what to watch for if this scenario is developing
- **Antifragile opportunity** — what a convex portfolio would look like to benefit from this scenario

**Important:** These scenarios are not predictions. They are structured thought experiments about the tails — designed to challenge your assumptions and reveal hidden vulnerabilities in your portfolio.
""")

        st.subheader("🎬 AI-Generated Stress Scenario Lab")
        st.markdown("The AI generates **forward-looking hypothetical scenarios** grounded in current fragility signals. "
                    "These are not predictions — they are structured analyses of how detected vulnerabilities could "
                    "amplify a shock into a crisis. Each scenario feeds directly into the portfolio stress engine above.")
        if not anthropic_key:
            st.warning("Enter your **Groq API key** in the sidebar (Step ②) to unlock this tab.")
        elif not GROQ_AVAILABLE:
            st.error("`groq` package not installed. Run: `pip install groq`")
        else:
            sb, sn = st.columns([2,1])
            with sb:
                run_s = st.button("🎬 Generate Stress Scenarios", type="primary")
            with sn:
                n_scen = st.selectbox("Scenarios to generate", [3,4,5], index=2)
            if run_s:
                with st.spinner("Claude is generating stress scenarios from current fragility signals…"):
                    st.session_state.scenarios = run_scenario_agent(
                        anthropic_key, st.session_state.frag,
                        st.session_state.market, st.session_state.portfolio, n_scen)
                    st.session_state.narrative = run_narrative_agent(
                        anthropic_key, st.session_state.frag,
                        st.session_state.scenarios, st.session_state.portfolio,
                        st.session_state.market)

            narr  = st.session_state.narrative
            scens = st.session_state.scenarios

            if narr:
                st.markdown("### Investment Committee Briefing")
                st.markdown(f'<div style="background:rgba(0,0,0,.25);border-left:4px solid #6366f1;'
                            f'padding:1.1rem;border-radius:8px;line-height:1.75">{narr}</div>',
                            unsafe_allow_html=True)
                st.divider()

            if scens:
                valid_sc = [s for s in scens if "error" not in s and "market_impacts" in s]
                # Portfolio impact chart
                if valid_sc:
                    sim2 = PortfolioSimulator(st.session_state.portfolio, st.session_state.prices)
                    ai_impacts = []
                    for sc3 in valid_sc:
                        mi = sc3.get("market_impacts",{})
                        ai_impacts.append(sim2.instant_shock(
                            equity=mi.get("equity_shock",-0.15),
                            bond=mi.get("bond_move",0.05),
                            gold=mi.get("gold_move",0.05),
                            hy=mi.get("hy_credit_shock",-0.08),
                            ig=mi.get("ig_credit_shock",-0.03),
                            name=sc3.get("name","Unnamed"),
                        ))
                    if ai_impacts:
                        st.subheader(f"{len(valid_sc)} AI-Generated Scenarios — Portfolio Impact")
                        st.plotly_chart(chart_scenario_impact(ai_impacts), use_container_width=True)

                # Scenario cards
                tc = {"FINANCIAL":"#e74c3c","MACRO":"#f39c12","GEOPOLITICAL":"#9b59b6",
                      "TECHNICAL":"#3498db","LIQUIDITY":"#e67e22","CONTAGION":"#c0392b"}
                pc2 = {"LOW":"#2ecc71","MODERATE":"#f39c12","ELEVATED":"#e74c3c"}
                for sc4 in scens:
                    if "error" in sc4:
                        st.error(f"Scenario error: {sc4['error']}")
                        continue
                    sev_icon = "⚫" if sc4.get("severity")=="CATASTROPHIC" else "🔴" if sc4.get("severity")=="EXTREME" else "🟠"
                    with st.expander(f"{sev_icon} **{sc4.get('name','Unnamed')}** | {sc4.get('type','?')} | Prob: {sc4.get('probability_label','?')}"):
                        sa, sb2 = st.columns([3,2])
                        with sa:
                            st.markdown(f"**Trigger:** {sc4.get('trigger','')}")
                            st.markdown(f"**Fragility amplification:** {sc4.get('fragility_amplification','')}")
                            if sc4.get("taleb_annotation"):
                                st.markdown(f'<div style="background:rgba(99,102,241,.1);border-left:3px solid #6366f1;'
                                            f'padding:.55rem .9rem;border-radius:5px;font-size:.84rem;margin:.4rem 0">'
                                            f'📖 {sc4["taleb_annotation"]}</div>', unsafe_allow_html=True)
                            st.markdown(f"**Portfolio vulnerability:** {sc4.get('portfolio_vulnerability','')}")
                            if sc4.get("antifragile_opportunity"):
                                st.markdown(f'<div style="background:rgba(46,204,113,.08);border-left:3px solid #2ecc71;'
                                            f'padding:.55rem .9rem;border-radius:5px;font-size:.84rem;margin:.4rem 0">'
                                            f'✅ <b>Antifragile opportunity:</b> {sc4["antifragile_opportunity"]}</div>',
                                            unsafe_allow_html=True)
                        with sb2:
                            mi2 = sc4.get("market_impacts",{})
                            st.markdown("**Market impacts (estimates)**")
                            for label, key, suf in [
                                ("Equity shock","equity_shock",""),("Bond move","bond_move",""),
                                ("Gold move","gold_move",""),("HY credit","hy_credit_shock",""),
                                ("VIX peak est.","vix_peak_estimate"," pts"),
                                ("Duration","duration_days"," days")]:
                                v = mi2.get(key)
                                if v is not None:
                                    st.markdown(f"**{label}:** {fp(v) if isinstance(v,float) and abs(v)<10 else f'{v}{suf}'}")
                            ews = sc4.get("early_warning_signals",[])
                            if ews:
                                st.markdown("**Early warnings:**")
                                for ew in ews[:3]:
                                    st.markdown(f"• {ew}")
                        if sc4.get("historical_analog"):
                            st.info(f"📚 **Historical analog:** {sc4['historical_analog']}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div style="text-align:center;color:#475569;font-size:.78rem">'
            'Fragility Detection Agent · EADA Business School · Master\'s Thesis<br>'
            'Inspired by Nassim Nicholas Taleb\'s <i>Antifragile</i>, <i>The Black Swan</i>, '
            '<i>Dynamic Hedging</i><br>'
            'Data: Yahoo Finance · FRED · AI: Groq (Llama 3.3 70B) · '
            '<b>pip install streamlit yfinance groq scipy plotly</b>'
            '</div>', unsafe_allow_html=True)
