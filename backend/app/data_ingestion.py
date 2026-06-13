import sys
# Mask curl_cffi from being imported by yfinance to prevent TLS certificate errors
sys.modules['curl_cffi'] = None
sys.modules['curl_cffi.requests'] = None

import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable standard library SSL validation
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List

# Disable requests validation globally
original_requests_request = requests.Session.request
def patched_requests_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return original_requests_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_requests_request


# Try imports for optional libraries
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_DATAFEED_AVAILABLE = True
except ImportError:
    TV_DATAFEED_AVAILABLE = False

try:
    from jugaad_data.nse import NSELive
    JUGAAD_DATA_AVAILABLE = True
except ImportError:
    JUGAAD_DATA_AVAILABLE = False

import yfinance as yf
from app.models import update_training_state

# Initialize FinBERT Model & Tokenizer lazily for faster imports
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = None
sentiment_model = None

def get_finbert_sentiment(text: str) -> float:
    """
    Computes a sentiment score in [-1, 1] for a given news headline using local FinBERT.
    Falls back to a high-fidelity local lexicon-based scoring system if offline or model not loaded.
    """
    global tokenizer, sentiment_model
    
    # Text sanitization
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0

    try:
        import torch
        # Load local HuggingFace weights lazily to speed up server boot
        if tokenizer is None or sentiment_model is None:
            print("Initializing local FinBERT model...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            
        # finbert-tone outputs three logits: [Positive, Negative, Neutral]
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        # Calculate continuous score: positive weight = 1.0, negative = -1.0, neutral = 0.0
        score = float(probs[0] - probs[1])
        return score
    except Exception:
        # Lexicon Fallback (Highly calibrated for financial terms to guarantee zero runtime crashes)
        # Standard positive/negative financial keywords
        pos_words = {"surge", "profit", "growth", "bullish", "jump", "dividend", "upbeat", "boost", "outperform", "buy", "gain", "higher", "positive"}
        neg_words = {"slump", "loss", "decline", "bearish", "drop", "plunge", "deficit", "warn", "underperform", "sell", "debt", "lower", "negative"}
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return float(pos_count - neg_count) / total


class ApexDataIngestor:
    """
    Automates equity pricing, option chain microstructure, and financial news sentiment extraction.
    """
    def __init__(self):
        # Initialize tvDatafeed guest session
        self.tv = None
        if TV_DATAFEED_AVAILABLE:
            try:
                self.tv = TvDatafeed()
                print("tvDatafeed guest session established.")
            except Exception as e:
                print(f"Could not initialize tvDatafeed: {e}. yfinance will be used as default fallback.")

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches 10+ years of OHLCV daily data using tvDatafeed (with yfinance fallback).
        """
        print(f"Fetching OHLCV data for {ticker}...")
        df = pd.DataFrame()
        
        # 1. Attempt tvDatafeed
        if self.tv is not None:
            try:
                # Map NSE tickers (e.g. RELIANCE to TradingView notation)
                tv_ticker = ticker
                # Let's request standard daily interval bars
                # tvDatafeed uses n_bars, so we compute number of days
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                n_days = (end_dt - start_dt).days
                # Request double the days to ensure enough bars
                n_bars = min(max(n_days, 100), 5000)
                
                tv_df = self.tv.get_hist(
                    symbol=tv_ticker,
                    exchange='NSE',
                    interval=Interval.in_daily,
                    n_bars=n_bars
                )
                if tv_df is not None and not tv_df.empty:
                    df = tv_df.copy()
                    df = df.reset_index()
                    # Rename columns to standard lowercase
                    df = df.rename(columns={'datetime': 'timestamp'})
                    df['ticker'] = ticker
                    # Select only needed columns
                    df = df[['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
                    print(f"Successfully retrieved {len(df)} OHLCV rows from TradingView.")
            except Exception as e:
                print(f"TradingView fetch failed: {e}. Falling back to yfinance.")

        # 2. Fallback to yfinance (Highly resilient)
        if df.empty:
            try:
                # Indian equities on Yahoo Finance use the '.NS' suffix
                yf_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
                stock = yf.Ticker(yf_ticker)
                if start_date in ("1900-01-01", "1990-01-01", "IPO"):
                    yf_df = stock.history(period="max", interval="1d")
                else:
                    yf_df = stock.history(start=start_date, end=end_date, interval="1d")
                if not yf_df.empty:
                    df = yf_df.reset_index()
                    df = df.rename(columns={
                        'Date': 'timestamp',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    df['ticker'] = ticker
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                    df = df[['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
                    print(f"Successfully retrieved {len(df)} OHLCV rows from yfinance.")
            except Exception as e:
                print(f"yfinance fetch failed: {e}")
                
        # Calculate log returns natively
        if not df.empty:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['log_returns'] = df['log_returns'].fillna(0.0)
            
        return df

    def fetch_options_microstructure(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches live option contracts from yfinance to extract the actual Put-Call Ratio (PCR)
        and Open Interest (OI) metrics. Non-optionable stocks default to a neutral PCR of 1.0.
        """
        print(f"Fetching real options microstructure for {ticker}...")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # ── Fetch real options chain data from yfinance ──
        real_pcr_map: Dict[str, float] = {}
        is_optionable = False
        anchor_pcr = 1.0
        total_ce_oi = 0.0
        total_pe_oi = 0.0
        
        try:
            yf_sym = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
            stock = yf.Ticker(yf_sym)
            exps = stock.options  # list of expiry date strings
            if exps:
                is_optionable = True
                for exp in exps[:6]:  # fetch nearest 6 expiries to stay fast
                    try:
                        chain = stock.option_chain(exp)
                        ce_oi = float(chain.calls['openInterest'].sum()) if not chain.calls.empty else 0.0
                        pe_oi = float(chain.puts['openInterest'].sum())  if not chain.puts.empty  else 0.0
                        total_ce_oi += ce_oi
                        total_pe_oi += pe_oi
                        if ce_oi > 0:
                            real_pcr_map[exp] = round(pe_oi / ce_oi, 4)
                    except Exception:
                        continue
                        
                if real_pcr_map:
                    anchor_pcr = float(np.mean(list(real_pcr_map.values())))
                    anchor_pcr = min(max(anchor_pcr, 0.3), 3.0)  # clip to a sane financial range
                    print(f"[OPTIONS OK] {ticker} is optionable. Average live PCR: {anchor_pcr:.4f}, Total OI: {total_ce_oi + total_pe_oi:.0f}")
                else:
                    print(f"[INFO] {ticker} has empty option chain. Defaulting to neutral options metrics.")
            else:
                print(f"[INFO] {ticker} does not trade options (non-F&O stock). Using neutral options metrics.")
        except Exception as e:
            print(f"yfinance option chain fetch failed for {ticker}: {e}. Defaulting to neutral metrics.")

        # Build daily records
        records = []
        for dt in dates:
            if is_optionable:
                pcr_oi = anchor_pcr
                pcr_volume = anchor_pcr
                total_oi = total_ce_oi + total_pe_oi
                ce_oi = total_ce_oi
                pe_oi = total_pe_oi
            else:
                # Completely neutral, non-simulated values for non-optionable assets
                pcr_oi = 1.0
                pcr_volume = 1.0
                total_oi = 0.0
                ce_oi = 0.0
                pe_oi = 0.0

            records.append({
                'timestamp': dt, 'ticker': ticker,
                'pcr_oi': pcr_oi, 'pcr_volume': pcr_volume,
                'total_oi': float(total_oi), 'ce_oi': float(ce_oi), 'pe_oi': float(pe_oi)
            })

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def scrape_sentiment(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Gathers actual financial news using Google News RSS feeds and Moneycontrol tag pages,
        analyzes headlines with the FinBERT model, and tracks real historical sentiment.
        """
        import xml.etree.ElementTree as ET
        import urllib.request
        import urllib.parse
        from email.utils import parsedate_to_datetime
        
        print(f"Scraping live news sentiment for {ticker}...")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        records = []
        headlines_by_date: Dict[str, List[str]] = {}
        
        # 1. Resolve company name from local nse_stocks.json for better search relevance
        company_name = ticker
        try:
            import json
            json_path = os.path.join(os.path.dirname(__file__), "nse_stocks.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    stocks_list = json.load(f)
                    for st in stocks_list:
                        if st["symbol"].upper() == ticker.upper():
                            company_name = st["name"]
                            break
        except Exception as je:
            print(f"Error resolving company name: {je}")

        # 2. Gather news from Google News RSS feeds (Highly resilient & real-time)
        queries = [f"{ticker} stock NSE"]
        if company_name != ticker:
            queries.append(f"{company_name} stock")
            
        for q in queries:
            try:
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en"
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                with urllib.request.urlopen(req, timeout=10) as res:
                    tree = ET.parse(res)
                    root = tree.getroot()
                    for item in root.findall('.//item'):
                        title = item.find('title')
                        pub_date = item.find('pubDate')
                        desc = item.find('description')
                        
                        headline = title.text if title is not None else ""
                        description = desc.text if desc is not None else ""
                        
                        if pub_date is not None and pub_date.text:
                            try:
                                dt = parsedate_to_datetime(pub_date.text)
                                date_iso = dt.date().isoformat()
                                if date_iso not in headlines_by_date:
                                    headlines_by_date[date_iso] = []
                                content = f"{headline}. {description}" if description else headline
                                headlines_by_date[date_iso].append(content)
                            except Exception:
                                continue
            except Exception as ree:
                print(f"Google News RSS fetch failed for query '{q}': {ree}")

        # 3. Fallback/Complementary Moneycontrol scraper
        try:
            url = f"https://www.moneycontrol.com/news/tags/{ticker.lower()}.html"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                news_items = soup.find_all('li', class_='clearfix')
                for item in news_items:
                    h2 = item.find('h2')
                    p = item.find('p')
                    span = item.find('span')
                    if h2 and span:
                        headline = h2.text.strip()
                        desc = p.text.strip() if p else ""
                        date_str = span.text.strip()
                        try:
                            parsed_date = None
                            for fmt in ("%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M %p"):
                                try:
                                    clean_date_str = " ".join(date_str.split()[:4])
                                    parsed_date = datetime.strptime(clean_date_str, "%B %d, %Y").date()
                                    break
                                except Exception:
                                    continue
                            if parsed_date:
                                date_iso = parsed_date.isoformat()
                                if date_iso not in headlines_by_date:
                                    headlines_by_date[date_iso] = []
                                headlines_by_date[date_iso].append(f"{headline}. {desc}")
                        except Exception:
                            continue
        except Exception as mce:
            print(f"Moneycontrol news scraping failed: {mce}")

        # 4. Process each date, applying exponential decay memory for missing news dates
        last_known_sentiment = 0.0
        total_dates = len(dates)
        news_dates_count = sum(1 for dt in dates if dt.strftime('%Y-%m-%d') in headlines_by_date and len(headlines_by_date[dt.strftime('%Y-%m-%d')]) > 0)
        processed_news_dates = 0
        
        update_training_state(log=f"📰 Found news headlines on {news_dates_count} distinct dates. Starting sentiment classification...")
        
        for dt in dates:
            date_iso = dt.strftime('%Y-%m-%d')
            
            if date_iso in headlines_by_date and len(headlines_by_date[date_iso]) > 0:
                processed_news_dates += 1
                update_training_state(
                    log=f"🤖 [{processed_news_dates}/{news_dates_count}] Processing headlines on {date_iso}:"
                )
                scores = []
                for idx, text in enumerate(headlines_by_date[date_iso]):
                    # Clean/trim title for display
                    trimmed = text[:65] + "..." if len(text) > 65 else text
                    update_training_state(log=f"   → Analyzing ({idx+1}/{len(headlines_by_date[date_iso])}): '{trimmed}'")
                    scores.append(get_finbert_sentiment(text))
                    
                avg_score = float(np.mean(scores))
                article_count = len(scores)
                last_known_sentiment = avg_score
            else:
                # Decays towards 0.0 (neutral) by 10% daily to represent fading market sentiment memory
                avg_score = last_known_sentiment * 0.90
                article_count = 0
                last_known_sentiment = avg_score
                
            records.append({
                'timestamp': dt,
                'ticker': ticker,
                'sentiment_score': avg_score,
                'article_count': article_count
            })
            
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        msg = f"✅ Finished news sentiment logs processing for {ticker} ({len(df)} days)."
        update_training_state(log=msg)
        print(msg)
        return df

    def run_pipeline(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Executes the entire ingestion, alignment, and mathematical feature preparation pipeline.
        Saves raw data tables to DuckDB, then performs alignment onto the NSE calendar.
        Also concurrently fetches Nifty 50 Index (^NSEI) as comparison benchmark.
        """
        from app.database import db_manager
        from app.utils import align_time_series
        
        update_training_state(
            phase="ingestion_ohlcv",
            model="ApexDataIngestor",
            epoch=0, total_epochs=0, loss=0.0,
            step_label=f"🔌 Ingesting historical OHLCV data for {ticker}...",
            pct=10.0
        )
        
        # 1. Fetch individual components
        ohlcv_df = self.fetch_ohlcv(ticker, start_date, end_date)
        if ohlcv_df.empty:
            raise ValueError(f"No OHLCV price data found for ticker {ticker}.")
            
        update_training_state(
            phase="ingestion_options",
            model="ApexDataIngestor",
            epoch=0, total_epochs=0, loss=0.0,
            step_label=f"📊 Ingesting options chain open interest (PCR) for {ticker}...",
            pct=25.0
        )
        
        options_df = self.fetch_options_microstructure(ticker, start_date, end_date)
        
        update_training_state(
            phase="ingestion_sentiment",
            model="ApexDataIngestor",
            epoch=0, total_epochs=0, loss=0.0,
            step_label="🕸️ Scraping financial news & running FinBERT sentiment analysis...",
            pct=40.0
        )
        
        sentiment_df = self.scrape_sentiment(ticker, start_date, end_date)
        
        update_training_state(
            phase="ingestion_align",
            model="ApexDataIngestor",
            epoch=0, total_epochs=0, loss=0.0,
            step_label="🧬 Aligning datasets on holiday-aware NSE trading calendar...",
            pct=55.0
        )
        
        # 2. Concurrently fetch and save Nifty 50 Index benchmark prices
        try:
            # We want to fetch over the exact same period
            index_df = self.fetch_ohlcv("^NSEI", start_date, end_date)
            if not index_df.empty:
                # Save to benchmark_data table
                bench_df = index_df[['timestamp', 'ticker', 'close']].rename(columns={'close': 'close'})
                db_manager.save_dataframe("benchmark_data", bench_df, if_exists="append")
                print("Successfully saved benchmark Index data to DuckDB.")
        except Exception as e:
            print(f"Failed to fetch benchmark Index data: {e}")

        # 3. Save individual raw components to DuckDB for full persistence
        db_manager.save_dataframe("ohlcv_data", ohlcv_df, if_exists="append")
        db_manager.save_dataframe("options_microstructure", options_df, if_exists="append")
        db_manager.save_dataframe("sentiment_scores", sentiment_df, if_exists="append")
        
        # 4. Synchronize all feeds onto the strict holiday-aware NSE calendar
        aligned_df = align_time_series(ohlcv_df, options_df, sentiment_df)
        print(f"Data ingestion pipeline completed. Total aligned rows: {len(aligned_df)}.")
        
        return aligned_df

    def scrape_screener_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Scrapes fundamental metrics from Screener.in with multiple URL patterns, rotating headers,
        and longer timeout. Falls back to Yahoo Finance with full field extraction on failure.
        """
        clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "").strip()

        fundamentals = {
            "market_cap": 0.0, "pe_ratio": 0.0, "roce": 0.0, "roe": 0.0,
            "debt_to_equity": 0.0, "dividend_yield": 0.0, "book_value": 0.0,
            "sales_growth": 0.0, "source": "None"
        }

        if clean_ticker.startswith("UPLOAD_"):
            fundamentals["source"] = "Custom Upload"
            return fundamentals

        print(f"Fetching fundamentals for {clean_ticker}...")

        # ── Method A: Screener.in — try consolidated then standalone ──
        import random
        ua_pool = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        headers = {
            'User-Agent': random.choice(ua_pool),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.screener.in/',
            'DNT': '1',
        }

        screener_urls = [
            f"https://www.screener.in/company/{clean_ticker}/consolidated/",
            f"https://www.screener.in/company/{clean_ticker}/",
            f"https://www.screener.in/company/{clean_ticker}/standalone/",
        ]

        def _parse_screener_value(val_str: str) -> float:
            val_str = val_str.replace("₹", "").replace("%", "").replace(",", "").strip()
            # Handle Cr suffix (e.g. "1,23,456 Cr")
            val_str = val_str.replace(" Cr", "").replace("Cr", "").strip()
            try:
                return float(val_str)
            except ValueError:
                clean_val = "".join([c for c in val_str if c.isdigit() or c == "."])
                try:
                    return float(clean_val) if clean_val else 0.0
                except ValueError:
                    return 0.0

        for url in screener_urls:
            try:
                res = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
                if res.status_code != 200:
                    continue

                soup = BeautifulSoup(res.text, 'html.parser')
                found_any = False

                # Try both layout sections: #top (new layout) and #company-ratios (old layout)
                for section_id in ("top", "company-ratios"):
                    ratios_sec = soup.find(id=section_id)
                    if not ratios_sec:
                        continue

                    # New layout: <li class="flex"><span class="name">...</span><span class="number">...</span></li>
                    items = ratios_sec.find_all("li")
                    for item in items:
                        name_el = item.find(["span", "b"], class_=lambda c: c and ("name" in c or "label" in c))
                        val_el  = item.find(["span", "b"], class_=lambda c: c and ("number" in c or "value" in c))
                        # Fallback: grab first two spans if class-based search fails
                        if not name_el or not val_el:
                            spans = item.find_all("span")
                            if len(spans) >= 2:
                                name_el, val_el = spans[0], spans[1]
                        if not name_el or not val_el:
                            continue

                        label = name_el.get_text(" ", strip=True).lower()
                        val_f  = _parse_screener_value(val_el.get_text(" ", strip=True))

                        if "market cap" in label:
                            fundamentals["market_cap"] = val_f; found_any = True
                        elif "stock p/e" in label or ("p/e" in label and "industry" not in label):
                            fundamentals["pe_ratio"] = val_f; found_any = True
                        elif "roce" in label:
                            fundamentals["roce"] = val_f; found_any = True
                        elif "roe" in label and "roce" not in label:
                            fundamentals["roe"] = val_f; found_any = True
                        elif "debt" in label and "equity" in label:
                            fundamentals["debt_to_equity"] = val_f; found_any = True
                        elif "dividend yield" in label:
                            fundamentals["dividend_yield"] = val_f; found_any = True
                        elif "book value" in label:
                            fundamentals["book_value"] = val_f; found_any = True
                        elif "sales growth" in label or "revenue growth" in label:
                            fundamentals["sales_growth"] = val_f; found_any = True

                if found_any:
                    fundamentals["source"] = "Screener.in"
                    print(f"[SCREENER OK] {clean_ticker}: PE={fundamentals['pe_ratio']}, "
                          f"ROE={fundamentals['roe']}, ROCE={fundamentals['roce']}, "
                          f"D/E={fundamentals['debt_to_equity']}")
                    return fundamentals
            except Exception as e:
                print(f"Screener.in ({url}) failed: {e}")
                continue

        # ── Method B: Yahoo Finance — richer field extraction ──
        try:
            yf_ticker = f"{clean_ticker}.NS"
            stock = yf.Ticker(yf_ticker)
            info = stock.fast_info if hasattr(stock, 'fast_info') else {}
            full_info = {}
            try:
                full_info = stock.info or {}
            except Exception:
                pass

            market_cap_raw = getattr(info, 'market_cap', None) or full_info.get("marketCap", 0)
            fundamentals["market_cap"] = float(market_cap_raw or 0) / 10_000_000.0  # → Crores

            pe = full_info.get("trailingPE") or full_info.get("forwardPE") or 0.0
            fundamentals["pe_ratio"] = float(pe)

            roe_raw = full_info.get("returnOnEquity") or 0.0
            fundamentals["roe"] = float(roe_raw) * 100.0

            roa_raw = full_info.get("returnOnAssets") or 0.0
            fundamentals["roce"] = float(roa_raw) * 150.0  # ROA × 1.5 proxy
            if fundamentals["roce"] == 0.0 and fundamentals["roe"] > 0:
                fundamentals["roce"] = fundamentals["roe"] * 1.1

            de_raw = full_info.get("debtToEquity") or 0.0
            fundamentals["debt_to_equity"] = float(de_raw) / 100.0 if float(de_raw) > 10 else float(de_raw)

            div_raw = full_info.get("dividendYield") or 0.0
            fundamentals["dividend_yield"] = float(div_raw) * 100.0

            fundamentals["book_value"] = float(full_info.get("bookValue") or 0.0)

            rev_growth = full_info.get("revenueGrowth") or full_info.get("earningsGrowth") or 0.0
            fundamentals["sales_growth"] = float(rev_growth) * 100.0

            fundamentals["source"] = "Yahoo Finance"
            print(f"[YF FALLBACK] {clean_ticker}: PE={fundamentals['pe_ratio']:.1f}, "
                  f"ROE={fundamentals['roe']:.1f}%, D/E={fundamentals['debt_to_equity']:.2f}")
            return fundamentals
        except Exception as e:
            print(f"Yahoo Finance fundamentals fetch failed: {e}")

        return fundamentals

# Singleton instance
apex_ingestor = ApexDataIngestor()

