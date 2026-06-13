import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Union, Optional

# Official list of NSE & BSE (Indian Stock Market) Trading Holidays
# Standard trading holidays include Republic Day, Holi, Good Friday, Eid, Independence Day, Gandhi Jayanti, Diwali, Christmas, etc.
MARKET_HOLIDAYS = {
    # 2023
    date(2023, 1, 26),   # Republic Day
    date(2023, 3, 7),    # Holi
    date(2023, 3, 30),   # Ram Navami
    date(2023, 4, 4),    # Mahavir Jayanti
    date(2023, 4, 7),    # Good Friday
    date(2023, 4, 14),   # Ambedkar Jayanti
    date(2023, 5, 1),    # Maharashtra Day
    date(2023, 6, 29),   # Bakri Id
    date(2023, 8, 15),   # Independence Day
    date(2023, 9, 19),   # Ganesh Chaturthi
    date(2023, 10, 2),   # Gandhi Jayanti
    date(2023, 10, 24),  # Dussehra
    date(2023, 11, 14),  # Diwali Balipratipada
    date(2023, 11, 27),  # Gurunanak Jayanti
    date(2023, 12, 25),  # Christmas
    
    # 2024
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 8),    # Mahashivratri
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 11),   # Ramzan Id
    date(2024, 4, 17),   # Ram Navami
    date(2024, 5, 1),    # Maharashtra Day
    date(2024, 6, 17),   # Bakri Id
    date(2024, 7, 17),   # Muharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Gandhi Jayanti
    date(2024, 11, 1),   # Diwali Laxmi Puja
    date(2024, 11, 15),  # Gurunanak Jayanti
    date(2024, 12, 25),  # Christmas

    # 2025
    date(2025, 1, 26),   # Republic Day
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Ramzan Id
    date(2025, 4, 10),   # Mahavir Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 9, 5),    # Id-E-Milad
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 20),  # Dussehra
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas

    # 2026
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Ram Navami
    date(2026, 3, 31),   # Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Gurunanak Jayanti
    date(2026, 12, 25),  # Christmas
}

# 2027 holidays (estimated/scheduled)
HOLIDAYS_2027 = {
    date(2027, 1, 26),   # Republic Day
    date(2027, 3, 6),    # Maha Shivaratri
    date(2027, 3, 10),   # Ramzan Id
    date(2027, 3, 22),   # Holi
    date(2027, 3, 26),   # Good Friday
    date(2027, 4, 14),   # Ambedkar Jayanti
    date(2027, 4, 19),   # Mahavir Jayanti
    date(2027, 5, 1),    # Maharashtra Day (Sat)
    date(2027, 5, 16),   # Bakri Id (Sun)
    date(2027, 7, 16),   # Muharram
    date(2027, 8, 15),   # Independence Day (Sun)
    date(2027, 9, 6),    # Ganesh Chaturthi
    date(2027, 9, 15),   # Id-E-Milad
    date(2027, 10, 2),   # Gandhi Jayanti (Sat)
    date(2027, 10, 9),   # Dussehra (Sat)
    date(2027, 10, 29),  # Diwali Laxmi Pujan
    date(2027, 11, 1),   # Diwali Balipratipada
    date(2027, 11, 14),  # Gurunanak Jayanti (Sun)
    date(2027, 12, 25),  # Christmas (Sat)
}

def is_nse_holiday(dt: Union[datetime, date]) -> bool:
    """
    Checks if a given date is an official NSE/BSE trading holiday or weekend.
    """
    if isinstance(dt, datetime):
        dt = dt.date()
    
    # 0 = Monday, 5 = Saturday, 6 = Sunday
    if dt.weekday() >= 5:
        return True
    
    year = dt.year
    if year <= 2026:
        return dt in MARKET_HOLIDAYS
    elif year == 2027:
        return dt in HOLIDAYS_2027
    else:
        # Dynamic fallback for future years (2028+)
        # Republic Day (Jan 26), Ambedkar Jayanti (Apr 14), Maharashtra Day (May 1),
        # Independence Day (Aug 15), Gandhi Jayanti (Oct 2), Christmas (Dec 25)
        if dt.month == 1 and dt.day == 26:
            return True
        if dt.month == 4 and dt.day == 14:
            return True
        if dt.month == 5 and dt.day == 1:
            return True
        if dt.month == 8 and dt.day == 15:
            return True
        if dt.month == 10 and dt.day == 2:
            return True
        if dt.month == 12 and dt.day == 25:
            return True
        return False

def get_nse_trading_days(start_date: Union[str, datetime, date], end_date: Union[str, datetime, date]) -> pd.DatetimeIndex:
    """
    Generates a DatetimeIndex representing only valid NSE trading sessions (excludes weekends and holidays).
    """
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = [d for d in all_days if not is_nse_holiday(d.date())]
    return pd.DatetimeIndex(trading_days)

def align_time_series(
    ohlcv_df: pd.DataFrame, 
    options_df: Optional[pd.DataFrame] = None, 
    sentiment_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Aligns multiple disparate time-series datasets onto a strict, holiday-aware NSE business calendar.
    Ensures zero forward-looking bias by using proper forward-fills.
    
    Args:
        ohlcv_df: Primary equity data with a DatetimeIndex (or column named 'timestamp').
        options_df: Options PCR and Open Interest dataframe.
        sentiment_df: Sentiment scores dataframe.
        
    Returns:
        A chronologically aligned, merged Pandas DataFrame containing all synchronized features.
    """
    # Standardize column index
    if 'timestamp' in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.set_index('timestamp')
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)

    # Save original timestamps (with their time components, e.g. 09:15:00)
    # mapped by date for restoring at the end.
    original_timestamps = pd.Series(ohlcv_df.index, index=ohlcv_df.index.normalize())

    # Establish the calendar boundaries using normalized dates
    start_date = ohlcv_df.index.normalize().min()
    end_date = ohlcv_df.index.normalize().max()
    
    # Create the strict trading-day calendar (it will have 00:00:00 timestamps)
    nse_calendar = pd.DatetimeIndex([d for d in pd.date_range(start_date, end_date, freq='D') if not is_nse_holiday(d.date())])
    
    # Reindex OHLCV (normalized to 00:00:00) to fill any missing market sessions
    ohlcv_norm = ohlcv_df.copy()
    ohlcv_norm.index = ohlcv_norm.index.normalize()
    aligned_df = ohlcv_norm.reindex(nse_calendar)
    aligned_df.index.name = 'timestamp'
    
    # Forward-fill prices to handle occasional intra-day data gaps
    aligned_df[['open', 'high', 'low', 'close']] = aligned_df[['open', 'high', 'low', 'close']].ffill()
    aligned_df['volume'] = aligned_df['volume'].fillna(0.0)
    
    # Recompute log returns to ensure exact additive symmetry on the aligned series
    aligned_df['log_returns'] = np.log(aligned_df['close'] / aligned_df['close'].shift(1))
    aligned_df['log_returns'] = aligned_df['log_returns'].fillna(0.0)

    # 2. Merge Options Microstructure if available
    if options_df is not None and not options_df.empty:
        if 'timestamp' in options_df.columns:
            options_df = options_df.set_index('timestamp')
        options_df.index = pd.to_datetime(options_df.index).normalize()
        
        # Reindex option chain values to calendar and ffill
        options_aligned = options_df.reindex(nse_calendar).ffill().fillna(0.0)
        
        # Merge options features
        aligned_df = aligned_df.join(options_aligned[['pcr_oi', 'pcr_volume', 'total_oi', 'ce_oi', 'pe_oi']], how='left')
        aligned_df['pcr_oi'] = aligned_df['pcr_oi'].fillna(1.0) # PCR neutral default is 1.0
        aligned_df['pcr_volume'] = aligned_df['pcr_volume'].fillna(1.0)
        aligned_df['total_oi'] = aligned_df['total_oi'].fillna(0.0)
        aligned_df['ce_oi'] = aligned_df['ce_oi'].fillna(0.0)
        aligned_df['pe_oi'] = aligned_df['pe_oi'].fillna(0.0)
    else:
        # Default options features if not provided
        aligned_df['pcr_oi'] = 1.0
        aligned_df['pcr_volume'] = 1.0
        aligned_df['total_oi'] = 0.0
        aligned_df['ce_oi'] = 0.0
        aligned_df['pe_oi'] = 0.0

    # 3. Merge Financial Sentiment if available
    if sentiment_df is not None and not sentiment_df.empty:
        if 'timestamp' in sentiment_df.columns:
            sentiment_df = sentiment_df.set_index('timestamp')
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        
        # Fill non-news days with 0.0 (neutral sentiment) rather than ffilling,
        # representing that no active sentiment event occurred on that session.
        # Note: scrape_sentiment already handles decay inside the daily series,
        # so reindexing with fillna(0.0) is correct here.
        sentiment_aligned = sentiment_df.reindex(nse_calendar).fillna(0.0)
        
        aligned_df = aligned_df.join(sentiment_aligned[['sentiment_score']], how='left')
        aligned_df['sentiment_score'] = aligned_df['sentiment_score'].fillna(0.0)
    else:
        aligned_df['sentiment_score'] = 0.0

    # Restore original timestamps (with their time components, e.g. 09:15:00)
    # For any dates not in original_timestamps (e.g. filled gaps), we construct a timestamp at 09:15:00.
    new_timestamps = []
    for d in aligned_df.index:
        orig = original_timestamps.get(d)
        if pd.notna(orig):
            new_timestamps.append(orig)
        else:
            new_timestamps.append(pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=15))
            
    aligned_df.index = pd.DatetimeIndex(new_timestamps)
    aligned_df = aligned_df.reset_index().rename(columns={'index': 'timestamp'})
    
    return aligned_df
