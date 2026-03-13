import os
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# EMR-6 Final
# ETF Momentum Rotation -> Stock Leader Rotation
# ------------------------------------------------------------
# 최종형 반영 사항
# 1) ETF 모멘텀 = 3M 60% + 1M 40%
# 2) ETF 중복 제거(corr cluster)
# 3) ETF holdings 후보 생성: cache -> CSV -> yfinance fallback
# 4) 종목 점수 = 추세 + 최근 모멘텀 + RS + 가속도
# 5) Top6 선정
# 6) 기존 보유 Top12 안이면 유지
# 7) 변동성 기반 비중 조절(동일비중보다 실전형)
# 8) 주간 자동 리밸런싱
#
# 필요 파일
# - data/emr6_etf_holdings_starter.csv
#
# 출력
# - data/emr6_current_portfolio.json
# - data/emr6_rebalance_log.csv
# - data/emr6_history.csv
# - data/emr6_holdings_cache/*.json
# ============================================================


# ============================================================
# 경로 / 환경변수
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
HOLDINGS_CACHE_DIR = os.path.join(DATA_DIR, "emr6_holdings_cache")

CURRENT_PORTFOLIO_FILE = os.path.join(DATA_DIR, "emr6_current_portfolio.json")
REBALANCE_LOG_FILE = os.path.join(DATA_DIR, "emr6_rebalance_log.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "emr6_history.csv")
ETF_HOLDINGS_FILE = os.path.join(DATA_DIR, "emr6_etf_holdings_starter.csv")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# ============================================================
# 전략 파라미터
# ============================================================
PRICE_HISTORY_PERIOD = "1y"
PRICE_INTERVAL = "1d"

ETF_TOP_N_RAW = 20
ETF_TOP_N_FINAL = 15

PORTFOLIO_SIZE = 6
HOLD_BUFFER_RANK = 12

ETF_CORR_LOOKBACK_DAYS = 126
ETF_CORR_THRESHOLD = 0.90

MIN_STOCK_PRICE = 10.0
MIN_STOCK_DOLLAR_VOLUME = 20_000_000
MAX_20D_DRAWDOWN_FILTER = -0.20
ETF_OVERHEAT_20D = 0.25

VOL_LOOKBACK_DAYS = 20
VOL_FLOOR = 0.015
VOL_CAP = 0.08

ETF_HOLDINGS_PER_ETF = 10
ETF_HOLDINGS_CACHE_MAX_AGE_DAYS = 21

TELEGRAM_MAX_CHARS = 3500


# ============================================================
# ETF 유니버스
# ============================================================
ETF_UNIVERSE: Dict[str, str] = {
    "XLK": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "MDY": "Mid Cap",
    "RSP": "Equal Weight S&P 500",
    "MTUM": "Momentum",
    "QUAL": "Quality",
    "USMV": "Low Volatility",
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors 2",
    "XSD": "Semiconductors Equal Weight",
    "IGV": "Software",
    "CLOU": "Cloud",
    "SKYY": "Cloud Infra",
    "BOTZ": "Robotics & AI",
    "HACK": "Cybersecurity",
    "DRIV": "Autonomous Vehicles",
    "PAVE": "Infrastructure",
    "ITA": "Aerospace Defense",
    "PKB": "Construction",
    "AIRR": "Industrial Innovation",
    "IYT": "Transportation",
    "XTN": "Transportation Equal Weight",
    "SEA": "Shipping",
    "KBE": "Banks",
    "KIE": "Insurance",
    "IAI": "Broker Dealers",
    "IPAY": "Digital Payments",
    "FINX": "Fintech",
    "XOP": "Oil & Gas Exploration",
    "OIH": "Oil Services",
    "VDE": "Energy Broad",
    "ICLN": "Clean Energy",
    "GRID": "Smart Grid",
    "XLU": "Utilities",
    "XRT": "Retail",
    "FDIS": "Consumer Disc Alt",
    "ONLN": "Online Retail",
    "IBUY": "E-commerce",
    "SOCL": "Social Media",
    "XME": "Metals & Mining",
    "KWEB": "China Internet",
    "EWJ": "Japan",
    "EWG": "Germany",
    "INDA": "India",
    "EWT": "Taiwan",
    "EZU": "Eurozone",
    "VGK": "Europe",
    "VWO": "Emerging Markets",
    "SPMO": "S&P Momentum",
    "VUG": "Large Growth",
    "VTV": "Large Value",
    "SCHG": "US Large Growth",
    "SCHD": "Dividend Quality",
    "IWF": "Russell 1000 Growth",
    "IWD": "Russell 1000 Value",
    "VO": "Midcap",
    "VB": "Smallcap",
}


# ============================================================
# 데이터 구조
# ============================================================
@dataclass
class ETFScore:
    ticker: str
    name: str
    close: float
    ret_1m: float
    ret_3m: float
    ma50: float
    ma200: float
    ret_20d: float
    score: float


@dataclass
class StockScore:
    ticker: str
    name: str
    source_etfs: str
    close: float
    dollar_volume_20: float
    ret_1m: float
    ret_3m: float
    ret_6m: float
    rs_3m_vs_spy: float
    acceleration: float
    volume_ratio: float
    vol_20d: float
    m_score: float
    r_score: float
    a_score: float
    v_score: float
    total_score: float
    rank: int = 0
    target_weight: float = 0.0


# ============================================================
# 유틸
# ============================================================
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HOLDINGS_CACHE_DIR, exist_ok=True)


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str) -> None:
    if not telegram_enabled():
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, data=payload, timeout=20)
    except Exception:
        pass


def send_telegram_chunked(title: str, lines: List[str], max_chars: int = TELEGRAM_MAX_CHARS) -> None:
    if not lines:
        send_telegram_message(title)
        return

    chunks: List[str] = []
    current = title

    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars:
            chunks.append(current)
            current = f"{title}\n{line}"
        else:
            current = candidate

    if current:
        chunks.append(current)

    total = len(chunks)
    if total == 1:
        send_telegram_message(chunks[0])
        return

    for i, chunk in enumerate(chunks, start=1):
        body_lines = chunk.split("\n")[1:] if "\n" in chunk else []
        send_telegram_message(f"{title} ({i}/{total})\n" + "\n".join(body_lines))


def safe_float(x) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def minmax_normalize(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() <= 1:
        return pd.Series([0.5] * len(v), index=v.index)

    vmin = v.min()
    vmax = v.max()
    if pd.isna(vmin) or pd.isna(vmax) or math.isclose(vmin, vmax):
        return pd.Series([0.5] * len(v), index=v.index)

    return (v - vmin) / (vmax - vmin)


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def infer_sector_bias_from_etfs(source_etfs: str) -> str:
    parts = [x.strip() for x in str(source_etfs).split(",") if x.strip()]
    if not parts:
        return "섹터 정보 없음"

    names = [ETF_UNIVERSE.get(x, x) for x in parts]
    return ", ".join(names[:3])


def action_label(status: str) -> str:
    mapping = {
        "KEEP": "보유",
        "IN": "신규 매수",
        "OUT": "제외",
    }
    return mapping.get(status, status)


def summarize_strengths(s: StockScore) -> str:
    strengths: List[str] = []

    if s.rank <= 3:
        strengths.append("상위 랭크")
    if s.ret_3m >= 0.20:
        strengths.append("3개월 모멘텀 강함")
    if s.ret_1m >= 0.10:
        strengths.append("최근 1개월 강함")
    if s.rs_3m_vs_spy >= 0.10:
        strengths.append("SPY 대비 RS 우위")
    if s.acceleration >= 0.05:
        strengths.append("가속도 양호")
    if s.vol_20d <= 0.03:
        strengths.append("변동성 안정")
    if s.volume_ratio >= 1.05:
        strengths.append("거래량 확장")

    if not strengths:
        strengths.append("상대 점수 우위")

    return ", ".join(strengths[:3])


def summarize_risks(s: StockScore) -> str:
    risks: List[str] = []

    if s.vol_20d >= 0.06:
        risks.append("변동성 높음")
    if s.acceleration < 0:
        risks.append("최근 가속 둔화")
    if s.rs_3m_vs_spy < 0.03:
        risks.append("RS 우위 제한적")
    if s.volume_ratio < 0.90:
        risks.append("거래량 둔화")
    if s.ret_1m < 0:
        risks.append("1개월 수익률 약함")

    if not risks:
        risks.append("특이 경고 없음")

    return ", ".join(risks[:2])


def build_reason_for_in(s: StockScore) -> str:
    reasons: List[str] = []

    if s.rank <= PORTFOLIO_SIZE:
        reasons.append("Top6 진입")
    if s.rs_3m_vs_spy >= 0.08:
        reasons.append("상대강도 우위")
    if s.acceleration >= 0.03:
        reasons.append("가속도 개선")
    if s.ret_3m >= 0.15:
        reasons.append("중기 모멘텀 강함")
    if s.vol_20d <= 0.035:
        reasons.append("변동성 안정")
    if s.source_etfs:
        reasons.append(f"상위 ETF 노출({s.source_etfs})")

    if not reasons:
        reasons.append("점수 우위")

    return ", ".join(reasons[:3])


def build_reason_for_keep(s: StockScore) -> str:
    reasons: List[str] = []

    if s.rank <= PORTFOLIO_SIZE:
        reasons.append("여전히 Top6")
    elif s.rank <= HOLD_BUFFER_RANK:
        reasons.append(f"보유 버퍼 유지(rank {s.rank})")

    if s.rs_3m_vs_spy >= 0.08:
        reasons.append("상대강도 유지")
    if s.ret_3m >= 0.12:
        reasons.append("모멘텀 유지")
    if s.acceleration >= 0:
        reasons.append("추세 훼손 없음")

    if not reasons:
        reasons.append("보유 조건 유지")

    return ", ".join(reasons[:3])


def build_reason_for_out(
    item: Dict,
    rank_map: Dict[str, int],
    score_map: Dict[str, float],
    final_positions: List[StockScore],
) -> str:
    ticker = str(item.get("ticker", "")).upper().strip()
    rank = rank_map.get(ticker)
    score = score_map.get(ticker)

    if rank is None:
        return "현재 후보 랭킹 이탈"
    if rank > HOLD_BUFFER_RANK:
        return f"보유 버퍼 이탈(rank {rank})"
    if rank > PORTFOLIO_SIZE:
        return f"Top6 밖으로 밀림(rank {rank})"

    weakest_final = None
    if final_positions:
        weakest_final = max(final_positions, key=lambda x: x.rank)

    if weakest_final and score is not None and score < weakest_final.total_score:
        return f"대체 후보 대비 점수 열위(score {score:.3f})"

    return "상대 점수 열위"


# ============================================================
# 기본 파일 초기화
# ============================================================
def ensure_base_files(as_of_date: Optional[str] = None) -> None:
    if not os.path.exists(CURRENT_PORTFOLIO_FILE):
        with open(CURRENT_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"as_of_date": as_of_date, "positions": []},
                f,
                ensure_ascii=False,
                indent=2,
            )

    if not os.path.exists(REBALANCE_LOG_FILE):
        pd.DataFrame(
            columns=[
                "date",
                "action",
                "ticker",
                "name",
                "rank",
                "score",
                "target_weight",
                "source_etfs",
            ]
        ).to_csv(REBALANCE_LOG_FILE, index=False, encoding="utf-8-sig")

    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "name",
                "rank",
                "score",
                "ret_1m",
                "ret_3m",
                "ret_6m",
                "rs_3m_vs_spy",
                "acceleration",
                "volume_ratio",
                "vol_20d",
                "target_weight",
                "source_etfs",
                "is_held",
            ]
        ).to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 가격 데이터
# ============================================================
def normalize_downloaded(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "date":
            rename_map[c] = "Date"
        elif cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().sort_values("Date").reset_index(drop=True)


def download_history(ticker: str, period: str = PRICE_HISTORY_PERIOD) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=PRICE_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_downloaded(raw)


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma20"] = out["Close"].rolling(20).mean()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma90"] = out["Close"].rolling(90).mean()
    out["ma200"] = out["Close"].rolling(200).mean()
    out["ret_20d"] = rolling_return(out["Close"], 20)
    out["ret_1m"] = rolling_return(out["Close"], 21)
    out["ret_3m"] = rolling_return(out["Close"], 63)
    out["ret_6m"] = rolling_return(out["Close"], 126)
    out["volume_ma20"] = out["Volume"].rolling(20).mean()
    out["volume_ma90"] = out["Volume"].rolling(90).mean()
    out["dollar_volume_20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    out["daily_ret"] = out["Close"].pct_change()
    out["vol_20d"] = out["daily_ret"].rolling(VOL_LOOKBACK_DAYS).std()
    return out


# ============================================================
# ETF 모멘텀
# ============================================================
def build_etf_scores() -> Tuple[List[ETFScore], Dict[str, pd.DataFrame]]:
    rows: List[ETFScore] = []
    price_map: Dict[str, pd.DataFrame] = {}

    for ticker, name in ETF_UNIVERSE.items():
        try:
            df = download_history(ticker)
            if df.empty or len(df) < 220:
                continue

            df = add_basic_indicators(df)
            price_map[ticker] = df

            row = df.iloc[-1]
            close = safe_float(row["Close"])
            ma50 = safe_float(row["ma50"])
            ma200 = safe_float(row["ma200"])
            ret_1m = safe_float(row["ret_1m"])
            ret_3m = safe_float(row["ret_3m"])
            ret_20d = safe_float(row["ret_20d"])

            if close is None or ma50 is None or ma200 is None or ret_1m is None or ret_3m is None or ret_20d is None:
                continue

            if close <= ma50:
                continue
            if close <= ma200:
                continue
            if ret_20d > ETF_OVERHEAT_20D:
                continue

            score = 0.60 * ret_3m + 0.40 * ret_1m

            rows.append(
                ETFScore(
                    ticker=ticker,
                    name=name,
                    close=close,
                    ret_1m=ret_1m,
                    ret_3m=ret_3m,
                    ma50=ma50,
                    ma200=ma200,
                    ret_20d=ret_20d,
                    score=score,
                )
            )
        except Exception:
            continue

    rows = sorted(rows, key=lambda x: x.score, reverse=True)
    return rows, price_map


def filter_etf_correlation(top_etfs: List[ETFScore], price_map: Dict[str, pd.DataFrame]) -> List[ETFScore]:
    selected: List[ETFScore] = []

    for candidate in top_etfs:
        keep = True
        c_df = price_map.get(candidate.ticker)
        if c_df is None or c_df.empty:
            continue

        c_ret = c_df["Close"].pct_change().dropna().tail(ETF_CORR_LOOKBACK_DAYS)

        for chosen in selected:
            s_df = price_map.get(chosen.ticker)
            if s_df is None or s_df.empty:
                continue

            s_ret = s_df["Close"].pct_change().dropna().tail(ETF_CORR_LOOKBACK_DAYS)
            merged = pd.concat([c_ret.rename("c"), s_ret.rename("s")], axis=1).dropna()
            if len(merged) < 60:
                continue

            corr = merged["c"].corr(merged["s"])
            if pd.notna(corr) and corr >= ETF_CORR_THRESHOLD:
                keep = False
                break

        if keep:
            selected.append(candidate)

        if len(selected) >= ETF_TOP_N_FINAL:
            break

    return selected


# ============================================================
# ETF holdings: CSV / cache / yfinance fallback
# ============================================================
def load_etf_holdings_csv() -> pd.DataFrame:
    if not os.path.exists(ETF_HOLDINGS_FILE):
        raise FileNotFoundError(
            f"필수 파일 없음: {ETF_HOLDINGS_FILE}\n"
            f"data 폴더 안에 emr6_etf_holdings_starter.csv 파일을 넣어야 합니다."
        )

    df = pd.read_csv(ETF_HOLDINGS_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"etf", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ETF holdings CSV 필수 컬럼 누락: {sorted(missing)}")

    if "name" not in df.columns:
        df["name"] = df["ticker"]
    if "weight_pct" not in df.columns:
        df["weight_pct"] = None

    df["etf"] = df["etf"].astype(str).str.upper().str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df["name"] = df["name"].astype(str).str.strip()
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce")

    df = df[(df["etf"] != "") & (df["ticker"] != "")]
    df = df.drop_duplicates(subset=["etf", "ticker"]).reset_index(drop=True)
    return df


def get_cache_path(etf_ticker: str) -> str:
    return os.path.join(HOLDINGS_CACHE_DIR, f"{etf_ticker.upper()}.json")


def is_cache_fresh(path: str, max_age_days: int = ETF_HOLDINGS_CACHE_MAX_AGE_DAYS) -> bool:
    if not os.path.exists(path):
        return False
    try:
        modified = datetime.fromtimestamp(os.path.getmtime(path))
        return modified >= (datetime.utcnow() - timedelta(days=max_age_days))
    except Exception:
        return False


def holdings_df_to_records(df: pd.DataFrame, etf_ticker: str) -> List[Dict]:
    if df is None or df.empty:
        return []

    temp = df.copy()
    if "etf" not in temp.columns:
        temp["etf"] = etf_ticker.upper()
    if "name" not in temp.columns:
        temp["name"] = temp["ticker"]

    temp["etf"] = temp["etf"].astype(str).str.upper().str.strip()
    temp["ticker"] = temp["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    temp["name"] = temp["name"].astype(str).str.strip()
    temp["weight_pct"] = pd.to_numeric(temp.get("weight_pct"), errors="coerce")

    temp = temp[(temp["etf"] != "") & (temp["ticker"] != "")]
    temp = temp.drop_duplicates(subset=["etf", "ticker"]).reset_index(drop=True)

    records: List[Dict] = []
    for row in temp.itertuples(index=False):
        records.append(
            {
                "etf": str(row.etf).upper().strip(),
                "ticker": str(row.ticker).upper().strip().replace(".", "-"),
                "name": str(row.name).strip() if row.name is not None else str(row.ticker),
                "weight_pct": None if pd.isna(row.weight_pct) else float(row.weight_pct),
            }
        )
    return records


def records_to_holdings_df(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])

    df = pd.DataFrame(records)
    for col in ["etf", "ticker", "name", "weight_pct"]:
        if col not in df.columns:
            df[col] = None

    df["etf"] = df["etf"].astype(str).str.upper().str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df["name"] = df["name"].astype(str).str.strip()
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce")
    df = df[(df["etf"] != "") & (df["ticker"] != "")]
    df = df.drop_duplicates(subset=["etf", "ticker"]).reset_index(drop=True)
    return df


def load_cached_etf_holdings(etf_ticker: str) -> pd.DataFrame:
    path = get_cache_path(etf_ticker)
    if not is_cache_fresh(path):
        return pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("holdings", [])
        df = records_to_holdings_df(records)
        if not df.empty:
            df["etf"] = etf_ticker.upper()
        return df
    except Exception:
        return pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])


def save_cached_etf_holdings(etf_ticker: str, df: pd.DataFrame, source: str) -> None:
    path = get_cache_path(etf_ticker)
    payload = {
        "etf": etf_ticker.upper(),
        "source": source,
        "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "holdings": holdings_df_to_records(df, etf_ticker),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def parse_yfinance_top_holdings(top_holdings, etf_ticker: str) -> pd.DataFrame:
    if top_holdings is None:
        return pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])

    df: Optional[pd.DataFrame] = None

    if isinstance(top_holdings, pd.DataFrame):
        df = top_holdings.copy()
    elif isinstance(top_holdings, pd.Series):
        df = top_holdings.to_frame().reset_index()
    else:
        try:
            df = pd.DataFrame(top_holdings)
        except Exception:
            df = None

    if df is None or df.empty:
        return pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])

    temp = df.copy().reset_index()

    cols = {str(c).lower(): c for c in temp.columns}

    ticker_col = None
    for candidate in ["symbol", "ticker", "holding", "index"]:
        if candidate in cols:
            ticker_col = cols[candidate]
            break

    name_col = None
    for candidate in ["name", "holdingname", "company", "description"]:
        if candidate in cols:
            name_col = cols[candidate]
            break

    weight_col = None
    for candidate in ["holding percent", "holdingpercent", "weight", "percent", "pct", "% assets", "asset_%"]:
        if candidate in cols:
            weight_col = cols[candidate]
            break

    if ticker_col is None:
        if "index" in temp.columns:
            ticker_col = "index"
        else:
            ticker_col = temp.columns[0]

    if name_col is None:
        name_col = ticker_col

    temp["etf"] = etf_ticker.upper()
    temp["ticker"] = temp[ticker_col].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    temp["name"] = temp[name_col].astype(str).str.strip()

    if weight_col is not None:
        temp["weight_pct"] = pd.to_numeric(temp[weight_col], errors="coerce")
    else:
        temp["weight_pct"] = None

    temp = temp[(temp["ticker"] != "") & (temp["ticker"] != "NAN")]

    if temp["weight_pct"].notna().any():
        temp = temp.sort_values(by=["weight_pct", "ticker"], ascending=[False, True], na_position="last")

    temp = temp[["etf", "ticker", "name", "weight_pct"]].drop_duplicates(subset=["etf", "ticker"]).reset_index(drop=True)
    return temp


def fetch_etf_holdings_from_yfinance(etf_ticker: str) -> pd.DataFrame:
    try:
        ticker_obj = yf.Ticker(etf_ticker)
        funds_data = None

        try:
            funds_data = ticker_obj.funds_data
        except Exception:
            try:
                funds_data = ticker_obj.get_funds_data()
            except Exception:
                funds_data = None

        if funds_data is None:
            return pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])

        top_holdings = getattr(funds_data, "top_holdings", None)
        df = parse_yfinance_top_holdings(top_holdings, etf_ticker)
        if df.empty:
            return df

        df = df.head(ETF_HOLDINGS_PER_ETF).copy()
        return df
    except Exception:
        return pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])


def build_candidate_sources_with_fallback(
    selected_etfs: List[ETFScore],
    holdings_df: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    candidate_sources: Dict[str, List[str]] = {}
    name_map: Dict[str, str] = {}
    source_map: Dict[str, str] = {}

    if holdings_df is None:
        holdings_df = pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])

    selected_map = {e.ticker: e for e in selected_etfs}
    csv_subset = holdings_df[holdings_df["etf"].isin(selected_map.keys())].copy()

    for etf in selected_map.keys():
        etf_df = pd.DataFrame(columns=["etf", "ticker", "name", "weight_pct"])
        source = "none"

        cached_df = load_cached_etf_holdings(etf)
        if not cached_df.empty:
            etf_df = cached_df.copy()
            source = "cache"
        else:
            temp_csv = csv_subset[csv_subset["etf"] == etf].copy()
            if not temp_csv.empty:
                temp_csv = temp_csv.sort_values(
                    by=["weight_pct", "ticker"],
                    ascending=[False, True],
                    na_position="last",
                ).head(ETF_HOLDINGS_PER_ETF)
                etf_df = temp_csv.copy()
                source = "csv"
            else:
                temp_yf = fetch_etf_holdings_from_yfinance(etf)
                if not temp_yf.empty:
                    etf_df = temp_yf.copy()
                    source = "yfinance"
                    save_cached_etf_holdings(etf, etf_df, source="yfinance")

        source_map[etf] = source

        if etf_df.empty:
            continue

        if "weight_pct" not in etf_df.columns:
            etf_df["weight_pct"] = None

        etf_df["etf"] = etf
        etf_df["ticker"] = etf_df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
        etf_df["name"] = etf_df["name"].astype(str).str.strip()
        etf_df["weight_pct"] = pd.to_numeric(etf_df["weight_pct"], errors="coerce")

        etf_df = etf_df[(etf_df["ticker"] != "") & (etf_df["ticker"] != "NAN")]
        etf_df = etf_df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        etf_df = etf_df.head(ETF_HOLDINGS_PER_ETF)

        for row in etf_df.itertuples(index=False):
            candidate_sources.setdefault(row.ticker, []).append(etf)
            if row.ticker not in name_map:
                clean_name = str(row.name).strip() if row.name is not None else ""
                name_map[row.ticker] = clean_name if clean_name else row.ticker

    return candidate_sources, name_map, source_map


# ============================================================
# 종목 스코어
# ============================================================
def build_stock_scores(
    candidate_sources: Dict[str, List[str]],
    name_map: Dict[str, str],
    spy_df: pd.DataFrame,
) -> List[StockScore]:
    spy_df = add_basic_indicators(spy_df)
    spy_row = spy_df.iloc[-1]
    spy_ret_3m = safe_float(spy_row["ret_3m"])
    if spy_ret_3m is None:
        spy_ret_3m = 0.0

    raw_rows: List[dict] = []

    for ticker, source_etfs in candidate_sources.items():
        try:
            df = download_history(ticker)
            if df.empty or len(df) < 220:
                continue

            df = add_basic_indicators(df)
            row = df.iloc[-1]

            close = safe_float(row["Close"])
            ma50 = safe_float(row["ma50"])
            ma200 = safe_float(row["ma200"])
            dollar_vol_20 = safe_float(row["dollar_volume_20"])
            ret_1m = safe_float(row["ret_1m"])
            ret_3m = safe_float(row["ret_3m"])
            ret_6m = safe_float(row["ret_6m"])
            ret_20d = safe_float(row["ret_20d"])
            volume_ma20 = safe_float(row["volume_ma20"])
            volume_ma90 = safe_float(row["volume_ma90"])
            vol_20d = safe_float(row["vol_20d"])

            if any(v is None for v in [
                close, ma50, ma200, dollar_vol_20, ret_1m, ret_3m,
                ret_6m, ret_20d, volume_ma20, volume_ma90, vol_20d
            ]):
                continue

            if close < MIN_STOCK_PRICE:
                continue
            if dollar_vol_20 < MIN_STOCK_DOLLAR_VOLUME:
                continue
            if close <= ma50:
                continue
            if close <= ma200:
                continue
            if ret_20d <= MAX_20D_DRAWDOWN_FILTER:
                continue

            rs_3m = ret_3m - spy_ret_3m
            acceleration = ret_1m - (ret_3m / 3.0)
            volume_ratio = (volume_ma20 / volume_ma90) if volume_ma90 and volume_ma90 > 0 else 1.0

            raw_rows.append(
                {
                    "ticker": ticker,
                    "name": name_map.get(ticker, ticker),
                    "source_etfs": ",".join(sorted(set(source_etfs))),
                    "close": close,
                    "dollar_volume_20": dollar_vol_20,
                    "ret_1m": ret_1m,
                    "ret_3m": ret_3m,
                    "ret_6m": ret_6m,
                    "rs_3m_vs_spy": rs_3m,
                    "acceleration": acceleration,
                    "volume_ratio": volume_ratio,
                    "vol_20d": vol_20d,
                    "m_raw": 0.65 * ret_3m + 0.35 * ret_1m,
                    "r_raw": rs_3m,
                    "a_raw": acceleration,
                    "v_raw": volume_ratio,
                }
            )
        except Exception:
            continue

    if not raw_rows:
        return []

    temp = pd.DataFrame(raw_rows)

    temp["m_score"] = minmax_normalize(temp["m_raw"])
    temp["r_score"] = minmax_normalize(temp["r_raw"])
    temp["a_score"] = minmax_normalize(temp["a_raw"])
    temp["v_score"] = minmax_normalize(temp["v_raw"])

    temp["total_score"] = (
        temp["m_score"] * 0.35
        + temp["r_score"] * 0.25
        + temp["a_score"] * 0.25
        + temp["v_score"] * 0.15
    )

    temp = temp.sort_values(["total_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
    temp["rank"] = range(1, len(temp) + 1)

    results: List[StockScore] = []
    for _, row in temp.iterrows():
        results.append(
            StockScore(
                ticker=str(row["ticker"]),
                name=str(row["name"]),
                source_etfs=str(row["source_etfs"]),
                close=float(row["close"]),
                dollar_volume_20=float(row["dollar_volume_20"]),
                ret_1m=float(row["ret_1m"]),
                ret_3m=float(row["ret_3m"]),
                ret_6m=float(row["ret_6m"]),
                rs_3m_vs_spy=float(row["rs_3m_vs_spy"]),
                acceleration=float(row["acceleration"]),
                volume_ratio=float(row["volume_ratio"]),
                vol_20d=float(row["vol_20d"]),
                m_score=float(row["m_score"]),
                r_score=float(row["r_score"]),
                a_score=float(row["a_score"]),
                v_score=float(row["v_score"]),
                total_score=float(row["total_score"]),
                rank=int(row["rank"]),
                target_weight=0.0,
            )
        )

    return results


# ============================================================
# 변동성 기반 비중
# ============================================================
def apply_volatility_weights(positions: List[StockScore]) -> List[StockScore]:
    if not positions:
        return positions

    inverse_vols: List[float] = []
    for p in positions:
        clipped_vol = min(max(p.vol_20d, VOL_FLOOR), VOL_CAP)
        inverse_vols.append(1.0 / clipped_vol)

    total = sum(inverse_vols)
    if total <= 0:
        equal_weight = 1.0 / len(positions)
        for p in positions:
            p.target_weight = equal_weight
        return positions

    for p, inv in zip(positions, inverse_vols):
        p.target_weight = inv / total

    return positions


# ============================================================
# 포트 상태
# ============================================================
def load_current_portfolio() -> Dict:
    if not os.path.exists(CURRENT_PORTFOLIO_FILE):
        return {"as_of_date": None, "positions": []}
    try:
        with open(CURRENT_PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"as_of_date": None, "positions": []}


def save_current_portfolio(as_of_date: str, positions: List[StockScore]) -> None:
    payload = {
        "as_of_date": as_of_date,
        "positions": [
            {
                "ticker": p.ticker,
                "name": p.name,
                "rank": p.rank,
                "score": round(p.total_score, 4),
                "target_weight": round(p.target_weight, 6),
                "source_etfs": p.source_etfs,
                "close": round(p.close, 2),
            }
            for p in positions
        ],
    }
    with open(CURRENT_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_rebalance_log(as_of_date: str, out_items: List[Dict], in_items: List[Dict], kept_items: List[Dict]) -> None:
    rows = []

    for item in out_items:
        rows.append(
            {
                "date": as_of_date,
                "action": "OUT",
                "ticker": item.get("ticker"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "target_weight": item.get("target_weight"),
                "source_etfs": item.get("source_etfs"),
            }
        )

    for item in in_items:
        rows.append(
            {
                "date": as_of_date,
                "action": "IN",
                "ticker": item.get("ticker"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "target_weight": item.get("target_weight"),
                "source_etfs": item.get("source_etfs"),
            }
        )

    for item in kept_items:
        rows.append(
            {
                "date": as_of_date,
                "action": "KEEP",
                "ticker": item.get("ticker"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "target_weight": item.get("target_weight"),
                "source_etfs": item.get("source_etfs"),
            }
        )

    df_new = pd.DataFrame(rows)
    if os.path.exists(REBALANCE_LOG_FILE):
        df_old = pd.read_csv(REBALANCE_LOG_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(REBALANCE_LOG_FILE, index=False, encoding="utf-8-sig")


def append_history(as_of_date: str, stock_scores: List[StockScore], held_tickers: List[str]) -> None:
    rows = []
    held_set = set(held_tickers)

    for s in stock_scores:
        rows.append(
            {
                "date": as_of_date,
                "ticker": s.ticker,
                "name": s.name,
                "rank": s.rank,
                "score": round(s.total_score, 6),
                "ret_1m": round(s.ret_1m, 6),
                "ret_3m": round(s.ret_3m, 6),
                "ret_6m": round(s.ret_6m, 6),
                "rs_3m_vs_spy": round(s.rs_3m_vs_spy, 6),
                "acceleration": round(s.acceleration, 6),
                "volume_ratio": round(s.volume_ratio, 6),
                "vol_20d": round(s.vol_20d, 6),
                "target_weight": round(s.target_weight, 6),
                "source_etfs": s.source_etfs,
                "is_held": 1 if s.ticker in held_set else 0,
            }
        )

    df_new = pd.DataFrame(rows)
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 리밸런싱
# ============================================================
def build_final_portfolio(stock_scores: List[StockScore]) -> Tuple[List[StockScore], List[Dict], List[Dict], List[Dict]]:
    current = load_current_portfolio()
    current_positions = current.get("positions", [])
    current_tickers = [str(x.get("ticker")).upper().strip() for x in current_positions]

    rank_map = {s.ticker: s.rank for s in stock_scores}
    score_map = {s.ticker: s.total_score for s in stock_scores}
    weight_map = {s.ticker: s.target_weight for s in stock_scores}
    obj_map = {s.ticker: s for s in stock_scores}

    kept: List[StockScore] = []
    for ticker in current_tickers:
        rank = rank_map.get(ticker)
        if rank is not None and rank <= HOLD_BUFFER_RANK and ticker in obj_map:
            kept.append(obj_map[ticker])

    final_positions: List[StockScore] = []
    for item in kept:
        if item.ticker not in {x.ticker for x in final_positions}:
            final_positions.append(item)

    for item in stock_scores[:PORTFOLIO_SIZE]:
        if len(final_positions) >= PORTFOLIO_SIZE:
            break
        if item.ticker not in {x.ticker for x in final_positions}:
            final_positions.append(item)

    if len(final_positions) < PORTFOLIO_SIZE:
        for item in stock_scores:
            if len(final_positions) >= PORTFOLIO_SIZE:
                break
            if item.ticker not in {x.ticker for x in final_positions}:
                final_positions.append(item)

    final_positions = sorted(final_positions, key=lambda x: x.rank)
    final_positions = apply_volatility_weights(final_positions)

    final_ticker_set = {x.ticker for x in final_positions}
    current_ticker_set = set(current_tickers)

    out_items = []
    for p in current_positions:
        t = str(p.get("ticker")).upper().strip()
        if t not in final_ticker_set:
            out_items.append(
                {
                    "ticker": t,
                    "name": p.get("name", t),
                    "rank": rank_map.get(t),
                    "score": score_map.get(t),
                    "target_weight": weight_map.get(t),
                    "source_etfs": p.get("source_etfs", ""),
                }
            )

    in_items = []
    for p in final_positions:
        if p.ticker not in current_ticker_set:
            in_items.append(
                {
                    "ticker": p.ticker,
                    "name": p.name,
                    "rank": p.rank,
                    "score": round(p.total_score, 4),
                    "target_weight": round(p.target_weight, 6),
                    "source_etfs": p.source_etfs,
                }
            )

    kept_items = []
    for p in final_positions:
        if p.ticker in current_ticker_set:
            kept_items.append(
                {
                    "ticker": p.ticker,
                    "name": p.name,
                    "rank": p.rank,
                    "score": round(p.total_score, 4),
                    "target_weight": round(p.target_weight, 6),
                    "source_etfs": p.source_etfs,
                }
            )

    return final_positions, out_items, in_items, kept_items


# ============================================================
# 리포트
# ============================================================
def summarize_holdings_sources(source_map: Dict[str, str]) -> str:
    counts = {"cache": 0, "csv": 0, "yfinance": 0, "none": 0}
    for v in source_map.values():
        counts[v] = counts.get(v, 0) + 1

    parts = []
    for key in ["cache", "csv", "yfinance", "none"]:
        if counts.get(key, 0) > 0:
            parts.append(f"{key} {counts[key]}")
    return " / ".join(parts) if parts else "-"


def build_report(
    as_of_date: str,
    top_etfs_raw: List[ETFScore],
    top_etfs_final: List[ETFScore],
    stock_scores: List[StockScore],
    final_positions: List[StockScore],
    out_items: List[Dict],
    in_items: List[Dict],
    kept_items: List[Dict],
    candidate_sources_count: int,
    holdings_source_map: Dict[str, str],
) -> str:
    lines: List[str] = []

    rank_map = {s.ticker: s.rank for s in stock_scores}
    score_map = {s.ticker: s.total_score for s in stock_scores}
    final_map = {s.ticker: s for s in final_positions}
    kept_set = {str(x.get("ticker")) for x in kept_items}

    lines.append("[EMR-6 주간 리밸런싱]")
    lines.append(f"기준일: {as_of_date}")
    lines.append("")
    lines.append("[요약]")
    lines.append(f"- 유지 {len(kept_items)} / 신규 {len(in_items)} / 제외 {len(out_items)}")
    lines.append(f"- 최종 ETF {len(top_etfs_final)}개 / 종목 후보 {len(stock_scores)}개")
    lines.append(f"- 후보 원천 종목 수 {candidate_sources_count}개")
    lines.append(f"- holdings source: {summarize_holdings_sources(holdings_source_map)}")
    if final_positions:
        sectors = unique_preserve_order([infer_sector_bias_from_etfs(s.source_etfs) for s in final_positions])
        lines.append(f"- 최종 포트 성격: {' / '.join(sectors[:3])}")
    lines.append("")

    lines.append("[이번 주 최종 포트폴리오]")
    if final_positions:
        for i, s in enumerate(final_positions, start=1):
            status = "KEEP" if s.ticker in kept_set else "IN"
            lines.append(f"{i}. {s.ticker} | {s.name} | {action_label(status)} | 비중 {fmt_pct(s.target_weight)}")
            lines.append(
                f"   - rank {s.rank} / score {s.total_score:.3f} / 1M {fmt_pct(s.ret_1m)} / 3M {fmt_pct(s.ret_3m)}"
            )
            lines.append(f"   - 강점: {summarize_strengths(s)}")
            lines.append(f"   - 주의: {summarize_risks(s)}")
            if status == "KEEP":
                lines.append(f"   - 이유: {build_reason_for_keep(s)}")
                lines.append("   - 실행: 기존 보유 유지")
            else:
                lines.append(f"   - 편입 이유: {build_reason_for_in(s)}")
                lines.append("   - 실행: 신규 매수 후보")
    else:
        lines.append("- 최종 포지션 없음")

    lines.append("")
    lines.append("[교체 요약]")

    lines.append("OUT")
    if out_items:
        for x in out_items:
            reason = build_reason_for_out(x, rank_map, score_map, final_positions)
            rank_txt = x["rank"] if x["rank"] is not None else "None"
            score_txt = fmt_num(x["score"], 3) if x["score"] is not None else "None"
            lines.append(
                f"- {x['ticker']} | {x['name']} | rank {rank_txt} | score {score_txt} | 이유: {reason}"
            )
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("IN")
    if in_items:
        for x in in_items:
            s = final_map.get(x["ticker"])
            reason = build_reason_for_in(s) if s is not None else "점수 우위"
            lines.append(
                f"- {x['ticker']} | {x['name']} | rank {x['rank']} | score {fmt_num(x['score'], 3)} | 비중 {fmt_pct(x['target_weight'])} | 이유: {reason}"
            )
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("[유지 종목]")
    if kept_items:
        for x in kept_items:
            s = final_map.get(x["ticker"])
            reason = build_reason_for_keep(s) if s is not None else "보유 조건 유지"
            lines.append(
                f"- {x['ticker']} | {x['name']} | rank {x['rank']} | score {fmt_num(x['score'], 3)} | 비중 {fmt_pct(x['target_weight'])} | 이유: {reason}"
            )
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("[상위 ETF]")
    if top_etfs_final:
        for i, e in enumerate(top_etfs_final[:8], start=1):
            source_txt = holdings_source_map.get(e.ticker, "none")
            lines.append(
                f"{i}. {e.ticker} | {e.name} | score {e.score:.3f} | 1M {fmt_pct(e.ret_1m)} | 3M {fmt_pct(e.ret_3m)} | holdings {source_txt}"
            )
    else:
        lines.append("- 없음")

    return "\n".join(lines)


def build_candidate_lines(stock_scores: List[StockScore]) -> List[str]:
    lines: List[str] = []
    separator = "────────────────"

    for s in stock_scores:
        lines.append(f"{s.rank}. {s.ticker} | {s.name}")
        lines.append(f"- score {s.total_score:.3f} | ETF {s.source_etfs}")
        lines.append(f"- M {s.m_score:.3f} | R {s.r_score:.3f} | A {s.a_score:.3f} | V {s.v_score:.3f}")
        lines.append(
            f"- 1M {fmt_pct(s.ret_1m)} | 3M {fmt_pct(s.ret_3m)} | RS {fmt_pct(s.rs_3m_vs_spy)} | ACC {fmt_pct(s.acceleration)} | VOL {fmt_pct(s.vol_20d)}"
        )
        lines.append(separator)
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    return lines


# ============================================================
# 메인
# ============================================================
def main() -> None:
    ensure_dirs()
    ensure_base_files()

    etf_scores_all, price_map = build_etf_scores()
    top_etfs_raw = etf_scores_all[:ETF_TOP_N_RAW]
    top_etfs_final = filter_etf_correlation(top_etfs_raw, price_map)

    spy_df = download_history("SPY")
    if spy_df.empty:
        raise RuntimeError("SPY 데이터 다운로드 실패")

    as_of_date = str(pd.to_datetime(spy_df.iloc[-1]["Date"]).date())
    ensure_base_files(as_of_date)

    if not top_etfs_final:
        save_current_portfolio(as_of_date, [])
        send_telegram_message(f"[EMR-6 주간 리밸런싱]\n기준일: {as_of_date}\nETF 후보가 없습니다.")
        print("ETF 후보가 없습니다.")
        return

    holdings_df = load_etf_holdings_csv()
    candidate_sources, name_map, holdings_source_map = build_candidate_sources_with_fallback(top_etfs_final, holdings_df)

    if not candidate_sources:
        save_current_portfolio(as_of_date, [])
        append_history(as_of_date, [], [])
        send_telegram_message(
            f"[EMR-6 주간 리밸런싱]\n기준일: {as_of_date}\nETF holdings 후보를 가져오지 못했습니다.\n"
            f"최종 ETF 수: {len(top_etfs_final)}\n"
            f"holdings source: {summarize_holdings_sources(holdings_source_map)}"
        )
        print("ETF holdings 후보를 가져오지 못했습니다.")
        return

    stock_scores = build_stock_scores(candidate_sources, name_map, spy_df)

    if not stock_scores:
        save_current_portfolio(as_of_date, [])
        append_history(as_of_date, [], [])
        send_telegram_message(
            f"[EMR-6 주간 리밸런싱]\n기준일: {as_of_date}\n종목 후보가 없습니다.\n"
            f"최종 ETF 수: {len(top_etfs_final)}\n"
            f"ETF holdings 종목 수: {len(candidate_sources)}\n"
            f"holdings source: {summarize_holdings_sources(holdings_source_map)}"
        )
        print("종목 후보가 없습니다.")
        return

    final_positions, out_items, in_items, kept_items = build_final_portfolio(stock_scores)

    save_current_portfolio(as_of_date, final_positions)
    append_rebalance_log(as_of_date, out_items, in_items, kept_items)
    append_history(as_of_date, stock_scores, [x.ticker for x in final_positions])

    report = build_report(
        as_of_date=as_of_date,
        top_etfs_raw=top_etfs_raw,
        top_etfs_final=top_etfs_final,
        stock_scores=stock_scores,
        final_positions=final_positions,
        out_items=out_items,
        in_items=in_items,
        kept_items=kept_items,
        candidate_sources_count=len(candidate_sources),
        holdings_source_map=holdings_source_map,
    )

    send_telegram_message(report)
    send_telegram_chunked("[EMR-6 후보 랭킹]", build_candidate_lines(stock_scores))
    print(report)


if __name__ == "__main__":
    main()
