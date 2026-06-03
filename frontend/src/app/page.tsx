'use client';

import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  TrendingUp, 
  Percent, 
  MessageSquare, 
  Cpu, 
  RefreshCw, 
  AlertTriangle,
  BookOpen,
  ShieldAlert,
  Play,
  RotateCcw,
  Upload,
  Search,
  Layers,
  LineChart,
  Calendar
} from 'lucide-react';
import ApexChart from '../components/ApexChart';

interface DataPoint {
  timestamp: string;
  close_raw: number;
  close_ffd: number;
  close_emd_smoothed: number;
  pcr_oi: number;
  sentiment_score: number;
  rolling_volatility: number;
  regime?: number;
  rsi?: number;
  macd?: number;
}

interface ForecastPoint {
  timestamp: string;
  forecast_close: number;
  lower_90: number;
  upper_90: number;
  lower_95: number;
  upper_95: number;
}

interface BacktestMetrics {
  sharpe_ratio: number;
  max_drawdown: number;
  mape: number;
  directional_accuracy: number;
  total_strategy_return: number;
}

interface TickerSuggestion {
  symbol: string;
  name: string;
  exchange: string;
}

export default function Dashboard() {
  const [ticker, setTicker] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [horizon, setHorizon] = useState(10);
  // Keep a ref so async callbacks always read the latest ticker value
  const tickerRef = useRef('');
  const [loading, setLoading] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Suggest search states
  const [suggestions, setSuggestions] = useState<TickerSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Tab selector states
  const [activeTab, setActiveTab] = useState<'forecast' | 'emd' | 'ffd' | 'hmm' | 'rsi' | 'macd'>('forecast');
  
  // Pipeline metrics state
  const [optimalD, setOptimalD] = useState<number | null>(null);
  const [regimeLabel, setRegimeLabel] = useState<string>('Bull/Low Vol');
  const [regimeState, setRegimeState] = useState<number>(0);
  const [chartData, setChartData] = useState<DataPoint[]>([]);
  const [forecasts, setForecasts] = useState<ForecastPoint[]>([]);
  const [backtestStats, setBacktestStats] = useState<BacktestMetrics | null>(null);
  const [macroStatus, setMacroStatus] = useState<string | null>(null);
  const [isIpoData, setIsIpoData] = useState(false);

  // Fundamental metrics state
  const [fundamentals, setFundamentals] = useState<any>(null);
  const [fundamentalRegime, setFundamentalRegime] = useState<string>('⚖️ STANDARD COMPOSITE');

  // Benchmarking and Ensemble state
  const [benchmark, setBenchmark] = useState<{ beta: number; alpha_annualized: number; correlation: number } | null>(null);
  const [ensembleWeights, setEnsembleWeights] = useState<{ tft: number; ridge: number; gbr: number; hw: number } | null>(null);

  // Advanced Institutional Workspace States
  const [workspaceTab, setWorkspaceTab] = useState<'cointegration' | 'var' | 'datatable' | 'fundamental_analytics'>('cointegration');
  const [sidebarTab, setSidebarTab] = useState<'backtest' | 'anomalies'>('backtest');

  // Cointegration Workspace
  const [cointTickers, setCointTickers] = useState('RELIANCE, TCS');
  const [cointLoading, setCointLoading] = useState(false);
  const [cointResult, setCointResult] = useState<any>(null);
  const [cointError, setCointError] = useState<string | null>(null);

  const runCointegration = async () => {
    setCointLoading(true);
    setCointError(null);
    setCointResult(null);
    try {
      const tickerList = cointTickers.split(',').map(t => t.trim().toUpperCase());
      if (tickerList.length < 2) throw new Error("Please enter at least 2 comma-separated tickers.");
      const queryParams = tickerList.map(t => `tickers=${t}`).join('&');
      const res = await fetch(`http://localhost:8000/api/math/cointegration?${queryParams}`, {
        method: 'POST'
      });
      if (!res.ok) {
        const errDetail = await res.json();
        throw new Error(errDetail.detail || "Cointegration testing failed.");
      }
      const data = await res.json();
      setCointResult(data.results);
    } catch (e: any) {
      setCointError(e.message);
    } finally {
      setCointLoading(false);
    }
  };

  // VAR Workspace
  const [varTarget, setVarTarget] = useState('RELIANCE');
  const [varMacro, setVarMacro] = useState('TCS');
  const [varLags, setVarLags] = useState(5);
  const [varLoading, setVarLoading] = useState(false);
  const [varResult, setVarResult] = useState<any>(null);
  const [varError, setVarError] = useState<string | null>(null);

  const runVarModel = async () => {
    setVarLoading(true);
    setVarError(null);
    setVarResult(null);
    try {
      const macroList = varMacro.split(',').map(t => t.trim().toUpperCase());
      if (!varTarget.trim()) throw new Error("Target ticker is required.");
      if (macroList.length === 0 || !macroList[0]) throw new Error("Please enter at least 1 macro index or comparison ticker.");
      const queryParams = macroList.map(t => `macro_tickers=${t}`).join('&');
      const res = await fetch(`http://localhost:8000/api/math/var?target_ticker=${varTarget.trim().toUpperCase()}&${queryParams}&lags=${varLags}`, {
        method: 'POST'
      });
      if (!res.ok) {
        const errDetail = await res.json();
        throw new Error(errDetail.detail || "VAR model fitting failed.");
      }
      const data = await res.json();
      setVarResult(data.results);
    } catch (e: any) {
      setVarError(e.message);
    } finally {
      setVarLoading(false);
    }
  };

  // Close suggestions dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Sync tickerRef whenever ticker state changes
  useEffect(() => { tickerRef.current = ticker; }, [ticker]);

  // Fetch search suggestions from local NSE DB (instant) with Yahoo Finance fallback
  useEffect(() => {
    if (searchQuery.trim().length < 1) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/stocks/search?q=${encodeURIComponent(searchQuery.trim())}`);
        if (res.ok) {
          const result = await res.json();
          setSuggestions(Array.isArray(result) ? result : []);
          setShowSuggestions(true);
        }
      } catch (err) {
        console.error('Suggestions fetch error:', err);
        setSuggestions([]);
      }
    }, 150);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const fetchAlignedData = async (targetTicker: string, pipelinePreview?: any[]) => {
    try {
      // If pipeline already returned preview data, use it directly (avoids extra DB round-trip)
      if (pipelinePreview && pipelinePreview.length > 0) {
        const mapped = pipelinePreview.map((r: any) => ({
          timestamp: r.timestamp,
          close_raw: r.close_raw ?? r.close ?? 0,
          close_ffd: r.close_ffd ?? 0,
          close_emd_smoothed: r.close_emd_smoothed ?? r.close_raw ?? r.close ?? 0,
          pcr_oi: r.pcr_oi ?? 0,
          sentiment_score: r.sentiment_score ?? 0,
          rolling_volatility: r.rolling_volatility ?? 0,
          regime: r.regime ?? 0,
          rsi: r.rsi ?? 50.0,
          macd: r.macd ?? 0.0,
        }));
        setChartData(mapped);
        if (mapped.length > 0) {
          const last = mapped[mapped.length - 1];
          setRegimeState(last.regime ?? 0);
          setRegimeLabel(last.regime === 1 ? 'Bear/High Vol' : 'Bull/Low Vol');
        }
        return;
      }
      // Fallback: fetch from DB preview endpoint
      const dataRes = await fetch(`http://localhost:8000/api/data/preview?ticker=${targetTicker.toUpperCase()}`);
      if (!dataRes.ok) throw new Error('Could not fetch aligned datasets.');
      const dataResult = await dataRes.json();
      const rows = (dataResult.data || []).map((r: any) => ({
        timestamp: r.timestamp,
        close_raw: r.close_raw ?? 0,
        close_ffd: r.close_ffd ?? 0,
        close_emd_smoothed: r.close_emd_smoothed ?? 0,
        pcr_oi: r.pcr_oi ?? 0,
        sentiment_score: r.sentiment_score ?? 0,
        rolling_volatility: r.rolling_volatility ?? 0,
        regime: r.regime ?? 0,
        rsi: r.rsi ?? 50.0,
        macd: r.macd ?? 0.0,
      }));
      setChartData(rows);
      if (rows.length > 0) {
        const last = rows[rows.length - 1];
        setRegimeState(last.regime ?? 0);
        setRegimeLabel(last.regime === 1 ? 'Bear/High Vol' : 'Bull/Low Vol');
      }
    } catch (e: any) {
      console.error('fetchAlignedData error:', e.message);
    }
  };

  const runPipeline = async (targetTicker?: string) => {
    const t = (targetTicker ?? tickerRef.current).trim().toUpperCase();
    if (!t) { setError('Please enter or select a stock ticker first.'); return; }
    setLoading(true);
    setError(null);
    setChartData([]);
    setForecasts([]);
    setBenchmark(null);
    setEnsembleWeights(null);
    try {
      const startDate = isIpoData ? '1900-01-01' : '';
      const res = await fetch(`http://localhost:8000/api/pipeline/run?ticker=${t}&start_date=${startDate}`, { method: 'POST' });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || 'Pipeline computation failed.');
      }
      const pipelineResult = await res.json();
      setOptimalD(pipelineResult.optimal_d);
      const resolvedTicker = (pipelineResult.ticker || t).toUpperCase();
      setTicker(resolvedTicker);
      setSearchQuery(resolvedTicker);
      tickerRef.current = resolvedTicker;
      
      if (pipelineResult.benchmark) {
        setBenchmark(pipelineResult.benchmark);
      }

      if (pipelineResult.fundamentals) {
        setFundamentals(pipelineResult.fundamentals);
      }
      
      // Use the preview data from pipeline response directly — no extra DB call needed
      await fetchAlignedData(resolvedTicker, pipelineResult.preview || []);
      setMacroStatus(`✓ ${resolvedTicker}: ${pipelineResult.data_count} sessions loaded. FFD d=${pipelineResult.optimal_d?.toFixed(3)}`);
    } catch (e: any) {
      setError(e.message || 'An unexpected pipeline error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const generateForecast = async (targetTicker?: string) => {
    const t = (targetTicker ?? tickerRef.current).trim().toUpperCase();
    if (!t) return;
    setPredicting(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/api/pipeline/predict?ticker=${t}&horizon_steps=${horizon}`, { method: 'POST' });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || 'Forecasting models execution failed.');
      }
      const predictResult = await res.json();
      setForecasts(predictResult.forecasts || []);
      setRegimeState(predictResult.current_regime ?? 0);
      setRegimeLabel(predictResult.regime_label ?? 'Bull/Low Vol');
      
      if (predictResult.ensemble_weights) {
        setEnsembleWeights(predictResult.ensemble_weights);
      }

      if (predictResult.fundamentals) {
        setFundamentals(predictResult.fundamentals);
      }
      if (predictResult.fundamental_regime) {
        setFundamentalRegime(predictResult.fundamental_regime);
      }
      
      setMacroStatus('Multi-Model Ensemble forecast path mapped with inverse-accuracy weights.');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setPredicting(false);
    }
  };

  const triggerBacktest = async (targetTicker?: string) => {
    const t = (targetTicker ?? tickerRef.current).trim().toUpperCase();
    if (!t) return;
    setBacktesting(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/api/pipeline/backtest?ticker=${t}&horizon_steps=${horizon}`, { method: 'POST' });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || 'Out-of-sample backtester failed.');
      }
      const backtestResult = await res.json();
      setBacktestStats(backtestResult.metrics);
      setMacroStatus('Out-of-sample walk-forward optimizer calibrated successfully.');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBacktesting(false);
    }
  };

  // Custom uploader
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    const baseName = file.name.split('.')[0];

    try {
      const res = await fetch(`http://localhost:8000/api/pipeline/upload?ticker_name=${baseName}`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errDetail = await res.json();
        throw new Error(errDetail.detail || "Custom dataset upload failed.");
      }

      const uploadResult = await res.json();
      const customTicker = uploadResult.ticker;

      setTicker(customTicker);
      setSearchQuery(customTicker);
      setOptimalD(uploadResult.optimal_d);
      setFundamentals(null);
      setFundamentalRegime('⚖️ STANDARD COMPOSITE');

      // Instantly run forecasting and backtesting metrics on uploaded set!
      await fetchAlignedData(customTicker);
      await generateForecast(customTicker);
      await triggerBacktest(customTicker);

      setMacroStatus(`Custom dataset [${file.name}] parsed and fitted immediately.`);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  // Do NOT auto-run on mount – wait for user to select a ticker
  // useEffect auto-run removed to fix "stuck on RELIANCE" bug

  const latestData = chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const prevData = chartData.length > 1 ? chartData[chartData.length - 2] : null;
  const priceChange = latestData && prevData 
    ? ((latestData.close_raw - prevData.close_raw) / prevData.close_raw) * 100 
    : 0;

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="logo-section">
          <h1>PROJECT APEX</h1>
          <p>Institutional-grade quantitative forecasting & risk envelopes</p>
        </div>
        
        <div className="controls-section">
          {/* Autocomplete Input Container */}
          <div style={{ position: 'relative' }} ref={dropdownRef}>
            <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '0 10px' }}>
              <Search size={16} color="var(--text-secondary)" style={{ marginRight: '6px' }} />
              <input 
                type="text" 
                className="ticker-input" 
                style={{ border: 'none', background: 'transparent', paddingLeft: '2px', textTransform: 'uppercase' }}
                value={searchQuery} 
                onChange={(e) => {
                  const val = e.target.value;
                  setSearchQuery(val);
                  if (val.trim().length >= 1) setShowSuggestions(true);
                  else setShowSuggestions(false);
                }}
                onFocus={() => { if (searchQuery.trim().length >= 1) setShowSuggestions(true); }}
                onKeyDown={async (e) => {
                  if (e.key === 'Enter' && searchQuery.trim().length >= 1) {
                    const t = searchQuery.trim().toUpperCase();
                    setTicker(t); tickerRef.current = t; setShowSuggestions(false);
                    await runPipeline(t); await generateForecast(t); await triggerBacktest(t);
                  }
                }}
                placeholder="SEARCH STOCK (E.G. DIXON)"
                disabled={loading || predicting || backtesting}
              />
            </div>
            {showSuggestions && suggestions.length > 0 && (
              <div className="suggestions-dropdown">
                {suggestions.map((item) => (
                  <div 
                    key={item.symbol} 
                    className="suggestion-item"
                    onClick={async () => {
                      const sym = item.symbol.trim().toUpperCase();
                      setTicker(sym);
                      setSearchQuery(sym);
                      tickerRef.current = sym;
                      setShowSuggestions(false);
                      setSuggestions([]);
                      await runPipeline(sym);
                      await generateForecast(sym);
                      await triggerBacktest(sym);
                    }}
                  >
                    <span className="symbol">{item.symbol}</span>
                    <span className="name">{item.name}</span>
                    <span className="exchange">{item.exchange}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* IPO Data toggle */}
          <button 
            className="action-btn"
            style={{ 
              background: isIpoData ? 'rgba(0, 242, 254, 0.1)' : 'rgba(255,255,255,0.01)',
              borderColor: isIpoData ? 'var(--accent-cyan)' : 'var(--border-color)',
              color: isIpoData ? 'var(--accent-cyan)' : 'var(--text-primary)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
            onClick={() => setIsIpoData(!isIpoData)}
            title="Download pricing records from its earliest available date since listing"
          >
            <Calendar size={14} />
            {isIpoData ? "EARLIEST DATA" : "STANDARD PRESET"}
          </button>

          {/* Custom File uploader */}
          <div style={{ position: 'relative' }}>
            <input 
              type="file" 
              id="custom-file-upload" 
              accept=".csv,.xlsx,.xls" 
              onChange={handleFileUpload} 
              style={{ display: 'none' }} 
              disabled={uploading}
            />
            <label 
              htmlFor="custom-file-upload" 
              className="action-btn"
              style={{ 
                cursor: 'pointer',
                background: 'rgba(255,255,255,0.02)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Upload size={14} />
              {uploading ? 'LOADING...' : 'UPLOAD CSV/XLSX'}
            </label>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '6px 12px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '700', letterSpacing: '0.5px' }}>HORIZON: {horizon}D</span>
            <input 
              type="range" 
              min="5" 
              max="30" 
              step="1"
              value={horizon}
              onChange={(e) => setHorizon(parseInt(e.target.value))}
              style={{ width: '80px', accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
              disabled={loading || predicting || backtesting}
            />
          </div>

          <button 
            className="action-btn"
            onClick={async () => {
              const t = searchQuery.trim().toUpperCase();
              if (!t) { setError('Please enter or select a stock ticker first.'); return; }
              setTicker(t); tickerRef.current = t;
              await runPipeline(t);
              await generateForecast(t);
              await triggerBacktest(t);
            }}
            disabled={loading || predicting || backtesting}
          >
            {loading ? 'COMPUTING...' : 'RUN PIPELINE'}
          </button>
        </div>
      </header>

      {/* Quantitative Methodology Framework */}
      <div style={{ display: 'flex', gap: '15px', background: 'rgba(0, 242, 254, 0.02)', border: '1px solid rgba(0, 242, 254, 0.1)', borderRadius: '12px', padding: '14px 20px', marginBottom: '25px', fontSize: '0.85rem', color: '#a3b2c5', lineHeight: '1.4' }}>
        <span style={{ fontWeight: '700', color: 'var(--accent-cyan)' }}>QUANTITATIVE METHODOLOGY FRAMEWORK:</span>
        <span>This institutional forecasting engine implements Fractional Differencing (FFD) to secure stationary features while preserving memory, utilizes Empirical Mode Decomposition (EMD) for signal noise filtering, applies a Gaussian Hidden Markov Model (HMM) to classify market volatility regimes, and compiles a dynamic weighted ensemble (TFT Attention, Robust Ridge, GBR, Holt-Winters) with Conformal calibration (MAPIE) to generate confidence envelopes. The ensembling weights and conformal prediction widths are dynamically calibrated based on the asset's fundamental regime (e.g. Growth, Value, Leverage) scraped from Screener.in/Yahoo Finance, blending long-term financial quality with short-term validation metrics.</span>
      </div>

      {error && (
        <div className="glass-panel" style={{ borderColor: 'var(--accent-red)', background: 'rgba(255, 23, 68, 0.05)', display: 'flex', gap: '10px', alignItems: 'center', margin: '0 0 20px 0' }}>
          <AlertTriangle color="var(--accent-red)" />
          <span style={{ color: 'var(--accent-red)', fontWeight: '600' }}>Error: {error}</span>
        </div>
      )}

      {/* KPI Cards row */}
      <div className="kpi-row">
        <div className="kpi-card">
          <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <TrendingUp size={16} color="var(--accent-cyan)" /> Raw Equity Price
          </span>
          <span className="kpi-value">
            {latestData ? `₹${latestData.close_raw.toFixed(2)}` : '—'}
          </span>
          <span className={`kpi-label ${priceChange >= 0 ? 'positive' : 'negative'}`} style={{ fontSize: '0.8rem', marginTop: '2px' }}>
            {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}% (1D)
          </span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ShieldAlert size={16} color={regimeState === 1 ? 'var(--accent-red)' : 'var(--accent-green)'} /> Classified Regime
          </span>
          <span className={`kpi-value ${regimeState === 1 ? 'negative' : 'positive'}`} style={{ fontSize: '1.4rem' }}>
            {regimeLabel}
          </span>
          <span className="kpi-label" style={{ fontSize: '0.8rem', marginTop: '2px' }}>
            Gaussian HMM transition logic
          </span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={16} color="var(--accent-green)" /> Conformal Bands (95%)
          </span>
          <span className="kpi-value cyan">
            {forecasts.length > 0 ? `±₹${((forecasts[0].upper_95 - forecasts[0].lower_95) / 2).toFixed(2)}` : '—'}
          </span>
          <span className="kpi-label" style={{ fontSize: '0.8rem', marginTop: '2px', color: 'var(--text-secondary)' }}>
            MAPIE calibrated residuals
          </span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Percent size={16} color="var(--accent-blue)" /> Put/Call Ratio (PCR)
          </span>
          <span className="kpi-value">
            {latestData ? latestData.pcr_oi.toFixed(2) : '—'}
          </span>
          <span className="kpi-label" style={{ fontSize: '0.8rem', marginTop: '2px', color: 'var(--text-secondary)' }}>
            Aggregated Near-Month contracts
          </span>
        </div>
      </div>

      {/* High-Density Benchmarking and Ensemble Allocator Row */}
      <div className="kpi-row" style={{ marginTop: '0px', marginBottom: '20px' }}>
        <div className="kpi-card" style={{ flex: 1.5, minWidth: '320px' }}>
          <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', fontWeight: '700', letterSpacing: '0.5px' }}>
            <Activity size={15} color="var(--accent-cyan)" /> Nifty 50 Systematic Risk Index
          </span>
          {benchmark ? (
            <div style={{ display: 'flex', gap: '15px', marginTop: '12px' }}>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontWeight: '600' }}>SYSTEMATIC BETA (β)</span>
                <span className="kpi-value" style={{ fontSize: '1.25rem' }}>{benchmark.beta.toFixed(2)}</span>
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', position: 'relative', marginTop: '4px' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${Math.min(100, (benchmark.beta / 2) * 100)}%`, background: benchmark.beta > 1.25 ? 'var(--accent-red)' : 'var(--accent-cyan)', borderRadius: '2px' }} />
                </div>
                <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '2px' }}>{benchmark.beta > 1.25 ? 'High Systematic Sens' : 'Low/Defensive Sens'}</span>
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontWeight: '600' }}>EXCESS ALPHA (α)</span>
                <span className="kpi-value" style={{ fontSize: '1.25rem', color: benchmark.alpha_annualized >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                  {benchmark.alpha_annualized >= 0 ? '+' : ''}{(benchmark.alpha_annualized * 100).toFixed(1)}%
                </span>
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', position: 'relative', marginTop: '4px' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${Math.min(100, Math.abs(benchmark.alpha_annualized) * 200)}%`, background: benchmark.alpha_annualized >= 0 ? 'var(--accent-green)' : 'var(--accent-red)', borderRadius: '2px' }} />
                </div>
                <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '2px' }}>Annualized relative drift</span>
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontWeight: '600' }}>INDEX CORRELATION (ρ)</span>
                <span className="kpi-value" style={{ fontSize: '1.25rem' }}>{benchmark.correlation.toFixed(2)}</span>
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', position: 'relative', marginTop: '4px' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${Math.min(100, Math.abs(benchmark.correlation) * 100)}%`, background: 'var(--accent-blue)', borderRadius: '2px' }} />
                </div>
                <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '2px' }}>Co-movement vector</span>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '15px' }}>Awaiting benchmark statistics...</div>
          )}
        </div>

        <div className="kpi-card" style={{ flex: 1.5, minWidth: '320px' }}>
          <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', fontWeight: '700', letterSpacing: '0.5px' }}>
            <Cpu size={15} color="var(--accent-green)" /> Multi-Model Allocation Weights
          </span>
          {ensembleWeights ? (
            <div style={{ marginTop: '12px' }}>
              {/* Horizontal Stacked Bar */}
              <div style={{ display: 'flex', height: '10px', borderRadius: '5px', overflow: 'hidden', background: 'rgba(255,255,255,0.05)', marginBottom: '12px' }}>
                <div style={{ width: `${ensembleWeights.tft * 100}%`, background: 'var(--accent-cyan)', transition: 'width 0.3s ease' }} title={`TFT: ${(ensembleWeights.tft * 100).toFixed(0)}%`} />
                <div style={{ width: `${ensembleWeights.ridge * 100}%`, background: '#ffeb3b', transition: 'width 0.3s ease' }} title={`Ridge: ${(ensembleWeights.ridge * 100).toFixed(0)}%`} />
                <div style={{ width: `${ensembleWeights.gbr * 100}%`, background: '#00e676', transition: 'width 0.3s ease' }} title={`GBR: ${(ensembleWeights.gbr * 100).toFixed(0)}%`} />
                <div style={{ width: `${ensembleWeights.hw * 100}%`, background: '#ff4081', transition: 'width 0.3s ease' }} title={`Holt-Winters: ${(ensembleWeights.hw * 100).toFixed(0)}%`} />
              </div>
              {/* Color Legend */}
              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-cyan)' }} />
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>TFT: <strong style={{ color: '#fff' }}>{(ensembleWeights.tft * 100).toFixed(0)}%</strong></span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ffeb3b' }} />
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Ridge: <strong style={{ color: '#fff' }}>{(ensembleWeights.ridge * 100).toFixed(0)}%</strong></span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#00e676' }} />
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>GBR: <strong style={{ color: '#fff' }}>{(ensembleWeights.gbr * 100).toFixed(0)}%</strong></span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ff4081' }} />
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>HW: <strong style={{ color: '#fff' }}>{(ensembleWeights.hw * 100).toFixed(0)}%</strong></span>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '15px' }}>Awaiting ensemble weight optimization...</div>
          )}
        </div>
      </div>

      {/* Institutional Fundamental Scorecard (Screener.in & Yahoo Finance) */}
      <div className="kpi-row" style={{ marginTop: '0px', marginBottom: '20px' }}>
        <div className="kpi-card" style={{ flex: 1, minWidth: '320px' }}>
          <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', fontWeight: '700', letterSpacing: '0.5px' }}>
            <Layers size={15} color="var(--accent-cyan)" /> Institutional Fundamental Scorecard (Screener.in)
          </span>
          {fundamentals ? (
            <div style={{ marginTop: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Fundamental Regime Calibration:</span>
                <span style={{ 
                  fontSize: '0.75rem', 
                  fontWeight: '700', 
                  color: fundamentalRegime.includes('QUALITY') ? 'var(--accent-cyan)' : fundamentalRegime.includes('VALUE') ? 'var(--accent-green)' : fundamentalRegime.includes('RISK') ? 'var(--accent-red)' : '#fff',
                  background: 'rgba(255, 255, 255, 0.05)',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  border: `1px solid ${fundamentalRegime.includes('QUALITY') ? 'rgba(0, 242, 254, 0.2)' : fundamentalRegime.includes('VALUE') ? 'rgba(0, 230, 118, 0.2)' : 'rgba(255, 23, 68, 0.2)'}`
                }}>
                  {fundamentalRegime}
                </span>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px', marginTop: '12px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(255,255,255,0.02)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>MARKET CAP (CR)</span>
                  <span style={{ fontSize: '0.95rem', color: '#fff', fontWeight: '600', marginTop: '2px' }}>
                    {fundamentals.market_cap ? `₹${fundamentals.market_cap.toLocaleString(undefined, {maximumFractionDigits: 0})} Cr` : '—'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(255,255,255,0.02)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>STOCK P/E RATIO</span>
                  <span style={{ fontSize: '0.95rem', color: '#fff', fontWeight: '600', marginTop: '2px' }}>
                    {fundamentals.pe_ratio ? fundamentals.pe_ratio.toFixed(2) : '—'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(255,255,255,0.02)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>ROCE / ROE</span>
                  <span style={{ fontSize: '0.95rem', color: 'var(--accent-green)', fontWeight: '600', marginTop: '2px' }}>
                    {fundamentals.roce ? `${fundamentals.roce.toFixed(1)}%` : '—'} / {fundamentals.roe ? `${fundamentals.roe.toFixed(1)}%` : '—'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(255,255,255,0.02)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>DEBT TO EQUITY</span>
                  <span style={{ fontSize: '0.95rem', color: fundamentals.debt_to_equity >= 1.5 ? 'var(--accent-red)' : '#fff', fontWeight: '600', marginTop: '2px' }}>
                    {fundamentals.debt_to_equity !== undefined ? fundamentals.debt_to_equity.toFixed(2) : '—'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(255,255,255,0.02)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>BOOK VALUE / DIV YIELD</span>
                  <span style={{ fontSize: '0.95rem', color: '#fff', fontWeight: '600', marginTop: '2px' }}>
                    ₹{fundamentals.book_value ? fundamentals.book_value.toFixed(1) : '—'} / {fundamentals.dividend_yield ? `${fundamentals.dividend_yield.toFixed(2)}%` : '—'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(255,255,255,0.02)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>SALES GROWTH (3Y)</span>
                  <span style={{ fontSize: '0.95rem', color: '#fff', fontWeight: '600', marginTop: '2px' }}>
                    {fundamentals.sales_growth ? `${fundamentals.sales_growth.toFixed(1)}%` : '—'}
                  </span>
                </div>
              </div>
              <div style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '10px', textAlign: 'right' }}>
                Source: {fundamentals.source} | Last updated: {fundamentals.updated_at ? fundamentals.updated_at.split('.')[0] : 'Never'}
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '15px' }}>Awaiting fundamental analysis. Run the pipeline to scrape Screener.in...</div>
          )}
        </div>
      </div>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Chart Column */}
        <div className="glass-panel" style={{ minHeight: '560px' }}>
          <div className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} /> Quantitative TSA Math Explorer
            </div>
            
            {/* Visualisation Tab Selectors */}
            <div className="tab-container" style={{ display: 'flex', background: 'rgba(255,255,255,0.02)', padding: '4px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <button 
                className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`}
                onClick={() => setActiveTab('forecast')}
                title="View predicted conformal envelopes and price targets"
              >
                <Activity size={14} /> FORECAST
              </button>
              <button 
                className={`tab-btn ${activeTab === 'emd' ? 'active' : ''}`}
                onClick={() => setActiveTab('emd')}
                title="View noise-filtered Empirical Mode Decomposition trend"
              >
                <Layers size={14} /> EMD CYCLES
              </button>
              <button 
                className={`tab-btn ${activeTab === 'ffd' ? 'active' : ''}`}
                onClick={() => setActiveTab('ffd')}
                title="View stationary Fractional Differencing series"
              >
                <LineChart size={14} /> FFD STATIONARY
              </button>
              <button 
                className={`tab-btn ${activeTab === 'hmm' ? 'active' : ''}`}
                onClick={() => setActiveTab('hmm')}
                title="View Hidden Markov Model regime transition boundaries"
              >
                <ShieldAlert size={14} /> HMM REGIMES
              </button>
              <button 
                className={`tab-btn ${activeTab === 'rsi' ? 'active' : ''}`}
                onClick={() => setActiveTab('rsi')}
                title="View Relative Strength Index (RSI) momentum cycles"
              >
                <Activity size={14} color="#e040fb" /> RSI
              </button>
              <button 
                className={`tab-btn ${activeTab === 'macd' ? 'active' : ''}`}
                onClick={() => setActiveTab('macd')}
                title="View Moving Average Convergence Divergence (MACD)"
              >
                <TrendingUp size={14} color="#00f2fe" /> MACD
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                className="action-btn" 
                style={{ padding: '0.4rem 1rem', fontSize: '0.8rem', background: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '4px' }}
                onClick={() => generateForecast()}
                disabled={predicting}
              >
                <Play size={12} /> {predicting ? 'CALIBRATING...' : 'FORECAST'}
              </button>
              <button 
                className="action-btn" 
                style={{ padding: '0.4rem 1rem', fontSize: '0.8rem', background: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '4px' }}
                onClick={() => triggerBacktest()}
                disabled={backtesting}
              >
                <RotateCcw size={12} /> {backtesting ? 'OPTIMIZING...' : 'BACKTEST'}
              </button>
            </div>
          </div>
          {/* Dynamic Tab Mathematical Context */}
          <div style={{ background: 'rgba(255, 255, 255, 0.01)', borderLeft: '3px solid var(--accent-cyan)', padding: '10px 15px', marginBottom: '16px', fontSize: '0.82rem', color: '#a3b2c5' }}>
            {activeTab === 'forecast' && (
              <span><strong>ENSEMBLE FORECAST ENVELOPES:</strong> Calibrates a dynamic weighted combination of self-attention (TFT), Ridge Regression, Gradient Boosting (GBR), and Holt-Winters (HW) models, overlaying 90% and 95% out-of-sample conformal margin risk boundaries.</span>
            )}
            {activeTab === 'emd' && (
              <span><strong>EMPIRICAL MODE DECOMPOSITION:</strong> Deconstructs non-stationary price signals into intrinsic mode functions (IMFs), separating low-frequency underlying structural trends from high-frequency white noise.</span>
            )}
            {activeTab === 'ffd' && (
              <span><strong>FRACTIONAL DIFFERENCING:</strong> Achieves a stationary mean-reverting series ($d \approx 0.35$ optimal threshold) while conserving maximum historic return memory compared to standard integer-differenced logs.</span>
            )}
            {activeTab === 'hmm' && (
              <span><strong>GAUSSIAN HIDDEN MARKOV MODEL:</strong> Maps market regimes based on price volatility and log-returns, plotting shifts between high-variance bear runs and low-variance bull cycles.</span>
            )}
            {activeTab === 'rsi' && (
              <span><strong>RELATIVE STRENGTH INDEX:</strong> Plots classical 14-period momentum indicators, highlighting local overbought threshold crossovers ($\ge 70$) and oversold panic regions ($\le 30$).</span>
            )}
            {activeTab === 'macd' && (
              <span><strong>MOVING AVERAGE CONVERGENCE DIVERGENCE:</strong> Analyzes exponential moving average crossovers (12 and 26-day MACD line vs zero line) to track trend strength and divergence dynamics.</span>
            )}
          </div>

          <div className="chart-container-wrapper">
            {loading || predicting || uploading ? (
              <div className="overlay-message">
                <div className="spinner"></div>
                <span>Executing Mathematical Pipelines (PTAR Attention Networks & MAPIE Envelopes)...</span>
              </div>
            ) : chartData.length > 0 ? (
              <ApexChart data={chartData} forecasts={forecasts} activeTab={activeTab} />
            ) : (
              <div className="overlay-message">
                <span>No dataset loaded. Enter a valid NSE symbol above or upload custom dataset.</span>
              </div>
            )}
          </div>
        </div>

        {/* Info Column */}
        <div>
          {/* Advanced Tabbed Quantitative Sidebar */}
          <div className="glass-panel" style={{ minHeight: '560px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '10px', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: '700', letterSpacing: '0.5px', color: 'var(--text-primary)' }}>
                QUANTITATIVE COGNITION
              </span>
              <div style={{ display: 'flex', background: 'rgba(255,255,255,0.02)', padding: '2px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <button 
                  className={`tab-btn ${sidebarTab === 'backtest' ? 'active' : ''}`}
                  onClick={() => setSidebarTab('backtest')}
                  style={{ padding: '4px 8px', fontSize: '0.7rem' }}
                >
                  METRICS
                </button>
                <button 
                  className={`tab-btn ${sidebarTab === 'anomalies' ? 'active' : ''}`}
                  onClick={() => setSidebarTab('anomalies')}
                  style={{ padding: '4px 8px', fontSize: '0.7rem' }}
                >
                  ANOMALIES
                </button>
              </div>
            </div>

            {sidebarTab === 'backtest' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%' }}>
                {/* Backtesting Stats Block */}
                <div>
                  <div className="panel-title" style={{ fontSize: '0.95rem', marginBottom: '8px' }}>
                    <Cpu size={16} /> Backtest Performance <span>{horizon}D Horizon</span>
                  </div>
                  {backtesting ? (
                    <div className="overlay-message" style={{ height: '140px' }}>
                      <div className="spinner" style={{ width: '25px', height: '25px' }}></div>
                      <span style={{ fontSize: '0.8rem' }}>Running Walk-Forward Optimization...</span>
                    </div>
                  ) : backtestStats ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '0.85rem' }}>
                      {/* Out-of-Sample Return */}
                      <div style={{ background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ color: 'var(--text-secondary)', fontWeight: '600' }}>Out-of-Sample Return</span>
                        <span className={backtestStats.total_strategy_return >= 0 ? 'positive' : 'negative'} style={{ fontWeight: '800', fontSize: '1.1rem' }}>
                          {backtestStats.total_strategy_return >= 0 ? '▲ ' : '▼ '}
                          {(backtestStats.total_strategy_return * 100).toFixed(2)}%
                        </span>
                      </div>

                      {/* Sharpe Ratio Visual Meter */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                          <span style={{ color: 'var(--text-secondary)', fontWeight: '600' }}>Out-of-Sample Sharpe Ratio</span>
                          <span style={{ color: 'var(--accent-cyan)', fontWeight: '700' }}>{backtestStats.sharpe_ratio.toFixed(2)}</span>
                        </div>
                        <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', position: 'relative', overflow: 'hidden', margin: '2px 0 2px 0' }}>
                          <div style={{ 
                            position: 'absolute', 
                            left: 0, 
                            top: 0, 
                            bottom: 0, 
                            width: `${Math.min(100, (Math.max(0, backtestStats.sharpe_ratio) / 3.0) * 100)}%`, 
                            background: backtestStats.sharpe_ratio > 2.0 ? 'var(--accent-green)' : backtestStats.sharpe_ratio > 1.0 ? '#ffeb3b' : 'var(--accent-red)',
                            borderRadius: '3px'
                          }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                          <span>0.0 (Poor)</span>
                          <span>1.5 (Good)</span>
                          <span>3.0+ (Institutional)</span>
                        </div>
                      </div>

                      {/* Maximum Drawdown Visual Meter */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                          <span style={{ color: 'var(--text-secondary)', fontWeight: '600' }}>Maximum Drawdown</span>
                          <span style={{ color: 'var(--accent-red)', fontWeight: '700' }}>{(backtestStats.max_drawdown * 100).toFixed(2)}%</span>
                        </div>
                        <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', position: 'relative', overflow: 'hidden', margin: '2px 0 2px 0' }}>
                          <div style={{ 
                            position: 'absolute', 
                            left: 0, 
                            top: 0, 
                            bottom: 0, 
                            width: `${Math.min(100, backtestStats.max_drawdown * 100 * 2.5)}%`, 
                            background: backtestStats.max_drawdown > 0.20 ? 'var(--accent-red)' : 'var(--accent-cyan)',
                            borderRadius: '3px'
                          }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                          <span>0%</span>
                          <span>20% (Warning)</span>
                          <span>40%+ (Critical)</span>
                        </div>
                      </div>

                      {/* Directional Accuracy Visual Meter */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                          <span style={{ color: 'var(--text-secondary)', fontWeight: '600' }}>Directional Accuracy</span>
                          <span style={{ color: 'var(--accent-green)', fontWeight: '700' }}>{(backtestStats.directional_accuracy * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', position: 'relative', overflow: 'hidden', margin: '2px 0 2px 0' }}>
                          <div style={{ 
                            position: 'absolute', 
                            left: 0, 
                            top: 0, 
                            bottom: 0, 
                            width: `${Math.min(100, backtestStats.directional_accuracy * 100)}%`, 
                            background: 'var(--accent-green)',
                            borderRadius: '3px'
                          }} />
                        </div>
                      </div>

                      {/* Conformal Uncertainty Band Width Widget */}
                      <div style={{ background: 'rgba(0, 242, 254, 0.02)', border: '1px solid rgba(0, 242, 254, 0.08)', borderRadius: '8px', padding: '10px', marginTop: '4px' }}>
                        <span style={{ fontSize: '0.7rem', color: 'var(--accent-cyan)', fontWeight: '700', display: 'block', letterSpacing: '0.5px' }}>CONFORMAL UNCERTAINTY BAND WIDTH</span>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Envelope Spread (95% CI)</span>
                          <span style={{ fontSize: '0.85rem', color: 'var(--text-primary)', fontWeight: '700' }}>
                            {forecasts.length > 0 ? `₹${((forecasts[0].upper_95 - forecasts[0].lower_95)).toFixed(2)}` : '—'}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Relative Width %</span>
                          <span style={{ fontSize: '0.75rem', color: forecasts.length > 0 && ((forecasts[0].upper_95 - forecasts[0].lower_95) / forecasts[0].forecast_close) > 0.15 ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: '600' }}>
                            {forecasts.length > 0 ? `${(((forecasts[0].upper_95 - forecasts[0].lower_95) / forecasts[0].forecast_close) * 100).toFixed(1)}%` : '—'}
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem', border: '1px dashed rgba(255,255,255,0.06)', borderRadius: '8px' }}>
                      Trigger BACKTEST to evaluate walk-forward performance.
                    </div>
                  )}
                </div>

                {/* Mathematical Foundations Block */}
                <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px' }}>
                  <div className="panel-title" style={{ fontSize: '0.95rem', marginBottom: '8px' }}>
                    <BookOpen size={16} /> Engine Foundations
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '4px' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Regime Detector</span>
                      <span style={{ color: 'var(--accent-red)', fontWeight: '600' }}>Gaussian HMM</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '4px' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Attention Engine</span>
                      <span style={{ color: 'var(--accent-cyan)', fontWeight: '600' }}>PTAR Attention</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '4px' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Quantification</span>
                      <span style={{ color: 'var(--accent-green)', fontWeight: '600' }}>MAPIE Conformal</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Spread Linkages</span>
                      <span style={{ color: 'var(--text-primary)' }}>Johansen test</span>
                    </div>
                  </div>
                </div>

                {/* Macro Status message */}
                <div style={{ marginTop: 'auto', padding: '8px', background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '8px', display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <RefreshCw size={12} color="var(--accent-green)" />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {macroStatus || 'Awaiting metrics calculation...'}
                  </span>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', height: '100%' }}>
                <div className="panel-title" style={{ fontSize: '0.95rem', marginBottom: '4px' }}>
                  <ShieldAlert size={16} color="var(--accent-cyan)" /> Real-Time Anomalies
                </div>
                
                {chartData.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto', maxHeight: '380px' }}>
                    {/* Anomaly 1: HMM Regime Shock */}
                    <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px' }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: '600', display: 'block' }}>HMM REGIME DETECTED</span>
                      <span style={{ fontSize: '0.85rem', fontWeight: '700', color: regimeState === 1 ? 'var(--accent-red)' : 'var(--accent-green)', display: 'block', marginTop: '2px' }}>
                        {regimeState === 1 ? '⚠ HIGH VOLATILITY BEAR REGIME' : '✓ LOW VOLATILITY BULL REGIME'}
                      </span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginTop: '4px' }}>
                        {regimeState === 1 ? 'High systematic variance. Enforce tight stop-loss boundaries.' : 'Stable upward drift. Multi-model dynamic forecasting validated.'}
                      </span>
                    </div>

                    {/* Anomaly 2: RSI Extremes */}
                    {(() => {
                      const latestRsi = chartData[chartData.length - 1]?.rsi ?? 50;
                      let rsiText = "Stable Momentum range. No overbought/oversold shocks.";
                      let rsiColor = "var(--text-primary)";
                      let rsiLabel = "✓ NORMAL MONEMTUM CONVERGENCE";
                      
                      if (latestRsi > 70) {
                        rsiLabel = "⚠ OVERBOUGHT EXTREME SHOCK";
                        rsiColor = "var(--accent-red)";
                        rsiText = `RSI at ${latestRsi.toFixed(1)} indicates atypical buying acceleration. Correction probability elevated.`;
                      } else if (latestRsi < 30) {
                        rsiLabel = "⚠ OVERSOLD PANIC CAPITULATION";
                        rsiColor = "var(--accent-green)";
                        rsiText = `RSI at ${latestRsi.toFixed(1)} indicates heavy selling exhaust. Strong potential turnaround region.`;
                      }
                      
                      return (
                        <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px' }}>
                          <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: '600', display: 'block' }}>RSI MOMENTUM ANOMALY</span>
                          <span style={{ fontSize: '0.85rem', fontWeight: '700', color: rsiColor, display: 'block', marginTop: '2px' }}>
                            {rsiLabel}
                          </span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginTop: '4px' }}>
                            {rsiText}
                          </span>
                        </div>
                      );
                    })()}

                    {/* Anomaly 3: Systematic Beta Shocks */}
                    <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px' }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: '600', display: 'block' }}>BENCHMARK SYSTEMATIC RISK</span>
                      <span style={{ fontSize: '0.85rem', fontWeight: '700', color: benchmark && benchmark.beta > 1.2 ? 'var(--accent-red)' : 'var(--accent-cyan)', display: 'block', marginTop: '2px' }}>
                        {benchmark ? (benchmark.beta > 1.25 ? '⚠ HIGH SYSTEMATIC AMPLITUDE' : '✓ DEFENSIVE RISK PROFILE') : 'AWAITING BENCHMARK RUN'}
                      </span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginTop: '4px' }}>
                        {benchmark ? `Asset beta is ${benchmark.beta.toFixed(2)}. ${benchmark.beta > 1.25 ? 'Highly reactive to Nifty 50 movement. High index-beta shock risk.' : 'Defensive counter-cyclical anchor. Insulated from indexing cascades.'}` : 'Run pipeline to acquire historical index-benchmarking stats.'}
                      </span>
                    </div>

                    {/* Anomaly 4: PCR Option Chain Sentiment */}
                    {(() => {
                      const latestPcr = chartData[chartData.length - 1]?.pcr_oi ?? 1.0;
                      let pcrLabel = "✓ BALANCED DERIVATIVES EXPOSURE";
                      let pcrColor = "var(--text-primary)";
                      let pcrText = "Standard put/call hedging patterns. Smooth institutional market clearance.";
                      
                      if (latestPcr > 1.45) {
                        pcrLabel = "⚠ BULLISH LIQUIDITY BUILDUP";
                        pcrColor = "var(--accent-green)";
                        pcrText = `PCR at ${latestPcr.toFixed(2)}: heavy institutional Put writing supporting local price floor.`;
                      } else if (latestPcr < 0.65) {
                        pcrLabel = "⚠ BEARISH HEDGING OUTFLOW";
                        pcrColor = "var(--accent-red)";
                        pcrText = `PCR at ${latestPcr.toFixed(2)}: massive Call buying/writing hedging against systematic correction.`;
                      }
                      
                      return (
                        <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px' }}>
                          <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: '600', display: 'block' }}>OPTIONS MICROSTRUCTURE</span>
                          <span style={{ fontSize: '0.85rem', fontWeight: '700', color: pcrColor, display: 'block', marginTop: '2px' }}>
                            {pcrLabel}
                          </span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginTop: '4px' }}>
                            {pcrText}
                          </span>
                        </div>
                      );
                    })()}
                  </div>
                ) : (
                  <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem', border: '1px dashed rgba(255,255,255,0.06)', borderRadius: '8px' }}>
                    Load a valid equity ticker to execute real-time anomaly checks.
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Advanced Quantitative Workspace */}
      <div className="glass-panel" style={{ marginTop: '2rem' }}>
        <h2 className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '16px', marginBottom: '20px' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} color="var(--accent-cyan)" /> ADVANCED INSTITUTIONAL MATHEMATICS LAB
          </span>
          <div className="tab-container" style={{ display: 'flex', background: 'rgba(255,255,255,0.02)', padding: '4px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <button 
              className={`tab-btn ${workspaceTab === 'cointegration' ? 'active' : ''}`}
              onClick={() => setWorkspaceTab('cointegration')}
              title="Test multivariate price linkages"
            >
              JOHANSEN COINTEGRATION
            </button>
            <button 
              className={`tab-btn ${workspaceTab === 'var' ? 'active' : ''}`}
              onClick={() => setWorkspaceTab('var')}
              title="Fit Vector Autoregressive models"
            >
              VECTOR AUTOREGRESSION (VAR)
            </button>
            <button 
              className={`tab-btn ${workspaceTab === 'datatable' ? 'active' : ''}`}
              onClick={() => setWorkspaceTab('datatable')}
              title="Inspect clean preprocessed numerical feature tables"
            >
              FEATURE REGISTRY TABLE
            </button>
            <button 
              className={`tab-btn ${workspaceTab === 'fundamental_analytics' ? 'active' : ''}`}
              onClick={() => setWorkspaceTab('fundamental_analytics')}
              title="Inspect fundamental calibration, ROCE/ROE gauges, and ensembling models justification"
            >
              FUNDAMENTAL ANALYTICS
            </button>
          </div>
        </h2>

        {/* Tab 1: Cointegration Test */}
        {workspaceTab === 'cointegration' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: '20px', marginBottom: '20px' }}>
              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: '600' }}>COINTEGRATION ASSETS:</span>
                <input 
                  type="text" 
                  className="ticker-input" 
                  style={{ width: '100%', textTransform: 'uppercase' }}
                  value={cointTickers} 
                  onChange={(e) => setCointTickers(e.target.value)}
                  placeholder="E.G. RELIANCE, TCS"
                />
                <button 
                  className="action-btn" 
                  style={{ width: '100%', padding: '0.6rem', fontSize: '0.9rem' }}
                  onClick={runCointegration}
                  disabled={cointLoading}
                >
                  {cointLoading ? 'TESTING...' : 'RUN COINTEGRATION'}
                </button>
                {cointError && <div style={{ color: 'var(--accent-red)', fontSize: '0.8rem', fontWeight: '600', marginTop: '6px' }}>Error: {cointError}</div>}
              </div>

              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.2rem', minHeight: '180px' }}>
                {cointResult ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <div style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '6px' }}>
                      Multivariate Johansen Test Stats (Eigen & Trace Vectors)
                    </div>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                          <th style={{ padding: '8px' }}>EIGENVALUE</th>
                          <th style={{ padding: '8px' }}>TRACE STAT</th>
                          <th style={{ padding: '8px' }}>MAX EIGEN STAT</th>
                          <th style={{ padding: '8px' }}>CRIT VAL (90%)</th>
                          <th style={{ padding: '8px' }}>CRIT VAL (95%)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {cointResult.eigenvalues.map((eig: number, idx: number) => (
                          <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                            <td style={{ padding: '8px', fontFamily: 'monospace' }}>{eig.toFixed(6)}</td>
                            <td style={{ padding: '8px', fontFamily: 'monospace', color: 'var(--accent-green)' }}>{cointResult.trace_stat[idx].toFixed(4)}</td>
                            <td style={{ padding: '8px', fontFamily: 'monospace' }}>{cointResult.max_eig_stat[idx].toFixed(4)}</td>
                            <td style={{ padding: '8px', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{cointResult.trace_crit[idx][0].toFixed(2)}</td>
                            <td style={{ padding: '8px', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{cointResult.trace_crit[idx][1].toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                      * If the Trace Statistic exceeds the critical values, we reject the null hypothesis of no-cointegration (assets share a long-term stationary mean-reverting spread).
                    </div>
                  </div>
                ) : (
                  <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                    Awaiting Johansen Cointegration parameters... Enter two or more loaded tickers (e.g. RELIANCE, TCS) and execute.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: VAR model */}
        {workspaceTab === 'var' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: '20px', marginBottom: '20px' }}>
              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: '600' }}>TARGET TICKER:</span>
                <input 
                  type="text" 
                  className="ticker-input" 
                  style={{ width: '100%', textTransform: 'uppercase' }}
                  value={varTarget} 
                  onChange={(e) => setVarTarget(e.target.value)}
                />
                
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: '600' }}>MACRO/SECTOR TICKERS:</span>
                <input 
                  type="text" 
                  className="ticker-input" 
                  style={{ width: '100%', textTransform: 'uppercase' }}
                  value={varMacro} 
                  onChange={(e) => setVarMacro(e.target.value)}
                  placeholder="E.G. TCS"
                />

                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: '600' }}>LAG ORDER (LAGS):</span>
                <input 
                  type="number" 
                  className="ticker-input" 
                  style={{ width: '100%' }}
                  value={varLags} 
                  onChange={(e) => setVarLags(Math.max(1, parseInt(e.target.value) || 5))}
                />

                <button 
                  className="action-btn" 
                  style={{ width: '100%', padding: '0.6rem', fontSize: '0.9rem', marginTop: '6px' }}
                  onClick={runVarModel}
                  disabled={varLoading}
                >
                  {varLoading ? 'CALIBRATING...' : 'FIT VAR MODEL'}
                </button>
                {varError && <div style={{ color: 'var(--accent-red)', fontSize: '0.8rem', fontWeight: '600', marginTop: '6px' }}>Error: {varError}</div>}
              </div>

              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.2rem', minHeight: '180px' }}>
                {varResult ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '6px', display: 'flex', justifyContent: 'space-between' }}>
                      <span>Vector Autoregressive Model Coefficients & P-Values</span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>AIC: {varResult.aic.toFixed(4)} | BIC: {varResult.bic.toFixed(4)} | Fitted Lags: {varResult.order}</span>
                    </div>
                    <div style={{ maxHeight: '180px', overflowY: 'auto', paddingRight: '4px' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                            <th style={{ padding: '6px' }}>VARIABLE RELATIONSHIP</th>
                            <th style={{ padding: '6px', textAlign: 'right' }}>LAG COEF p-VALUE</th>
                            <th style={{ padding: '6px', textAlign: 'right' }}>SIGNIFICANCE STATUS</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(varResult.pvalues).map(([relation, val]: [string, any]) => {
                            // val is a dictionary of coefficients p-values by equation
                            return Object.entries(val).map(([lagTerm, pVal]: [string, any]) => {
                              const p = parseFloat(pVal);
                              const isSig = p < 0.05;
                              return (
                                <tr key={`${relation}-${lagTerm}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                                  <td style={{ padding: '6px', fontFamily: 'monospace' }}>
                                    Equation <span style={{ color: 'var(--accent-blue)' }}>{relation}</span> ← {lagTerm}
                                  </td>
                                  <td style={{ padding: '6px', fontFamily: 'monospace', textAlign: 'right', color: isSig ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                                    {p.toFixed(6)}
                                  </td>
                                  <td style={{ padding: '6px', textAlign: 'right', fontWeight: '600', color: isSig ? 'var(--accent-green)' : 'rgba(255,255,255,0.2)' }}>
                                    {isSig ? '★ STATISTICALLY SIGNIFICANT' : 'INSIGNIFICANT'}
                                  </td>
                                </tr>
                              );
                            });
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                    Awaiting VAR model parameter selection... Enter target, sector assets, lag structures and fit.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Historical Registry Data Table */}
        {workspaceTab === 'datatable' && (
          <div>
            <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.2rem' }}>
              <div style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '8px', marginBottom: '12px' }}>
                Clean Stationary and Noise-Filtered Core Feature registry ({ticker})
              </div>
              
              {chartData.length > 0 ? (
                <div style={{ maxHeight: '240px', overflowY: 'auto', paddingRight: '4px' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)', position: 'sticky', top: 0, background: '#0c101c', zIndex: 1 }}>
                        <th style={{ padding: '8px' }}>SESSION DATE</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>CLOSE RAW (₹)</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>EMD SMOOTHED (₹)</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>FFD STATIONARY (d)</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>PUT/CALL RATIO (OI)</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>SENTIMENT</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartData.slice(-15).reverse().map((d, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                          <td style={{ padding: '8px', fontWeight: '500' }}>{d.timestamp}</td>
                          <td style={{ padding: '8px', fontFamily: 'monospace', textAlign: 'right' }}>₹{d.close_raw.toFixed(2)}</td>
                          <td style={{ padding: '8px', fontFamily: 'monospace', textAlign: 'right', color: 'var(--accent-cyan)' }}>₹{d.close_emd_smoothed.toFixed(2)}</td>
                          <td style={{ padding: '8px', fontFamily: 'monospace', textAlign: 'right', color: d.close_ffd >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{d.close_ffd.toFixed(4)}</td>
                          <td style={{ padding: '8px', fontFamily: 'monospace', textAlign: 'right' }}>{d.pcr_oi.toFixed(2)}</td>
                          <td style={{ padding: '8px', fontFamily: 'monospace', textAlign: 'right', color: d.sentiment_score > 0.02 ? 'var(--accent-green)' : d.sentiment_score < -0.02 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                            {d.sentiment_score.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  No historical features loaded. Please select a valid ticker or upload a custom CSV/XLSX.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 4: Fundamental Analytics and Gauge Visualizations */}
        {workspaceTab === 'fundamental_analytics' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '20px' }}>
              {/* Left Column: Financial Health Gauges */}
              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.2rem' }}>
                <div style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '8px', marginBottom: '16px' }}>
                  Institutional Quality Gauges ({ticker || 'No Ticker Loaded'})
                </div>
                {fundamentals ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                    {/* ROCE / ROE Performance Bar */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Return on Equity (ROE)</span>
                        <span style={{ fontWeight: '700', color: fundamentals.roe >= 15 ? 'var(--accent-green)' : '#fff' }}>
                          {fundamentals.roe ? `${fundamentals.roe.toFixed(1)}%` : '—'}
                        </span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ 
                          width: `${Math.min(100, Math.max(0, (fundamentals.roe ?? 0) * 3))}%`, 
                          background: (fundamentals.roe ?? 0) >= 15 ? 'var(--accent-green)' : 'var(--accent-cyan)',
                          height: '100%',
                          borderRadius: '4px'
                        }} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        <span>0%</span>
                        <span>15% (Quality Threshold)</span>
                        <span>30%+</span>
                      </div>
                    </div>

                    {/* Debt to Equity Leverage Gauge */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Debt to Equity Leverage Ratio</span>
                        <span style={{ fontWeight: '700', color: (fundamentals.debt_to_equity ?? 0) > 1.5 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                          {fundamentals.debt_to_equity !== undefined ? fundamentals.debt_to_equity.toFixed(2) : '—'}
                        </span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ 
                          width: `${Math.min(100, Math.max(0, (fundamentals.debt_to_equity ?? 0) * 50))}%`, 
                          background: (fundamentals.debt_to_equity ?? 0) > 1.5 ? 'var(--accent-red)' : (fundamentals.debt_to_equity ?? 0) > 0.8 ? '#ffeb3b' : 'var(--accent-green)',
                          height: '100%',
                          borderRadius: '4px'
                        }} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        <span>0.0 (Unleveraged)</span>
                        <span>1.0 (Moderate)</span>
                        <span>2.0+ (High Risk)</span>
                      </div>
                    </div>

                    {/* Sales Growth Growth Engine */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Compound 3-Year Sales Growth</span>
                        <span style={{ fontWeight: '700', color: (fundamentals.sales_growth ?? 0) >= 12 ? 'var(--accent-cyan)' : '#fff' }}>
                          {fundamentals.sales_growth ? `${fundamentals.sales_growth.toFixed(1)}%` : '—'}
                        </span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ 
                          width: `${Math.min(100, Math.max(0, (fundamentals.sales_growth ?? 0) * 3))}%`, 
                          background: (fundamentals.sales_growth ?? 0) >= 12 ? 'var(--accent-cyan)' : '#a3b2c5',
                          height: '100%',
                          borderRadius: '4px'
                        }} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        <span>0%</span>
                        <span>12% (Growth Baseline)</span>
                        <span>30%+</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                    Run the pipeline to load and analyze fundamental metrics.
                  </div>
                )}
              </div>

              {/* Right Column: Model Allocation Weights Justification */}
              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.2rem' }}>
                <div style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '8px', marginBottom: '16px' }}>
                  Model Allocation Rationale & Regimes
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '0.85rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Selected Regime:</span>
                    <span style={{ fontWeight: '800', color: 'var(--accent-cyan)' }}>{fundamentalRegime}</span>
                  </div>

                  <div style={{ color: 'var(--text-secondary)', lineHeight: '1.4', fontSize: '0.8rem', background: 'rgba(0, 242, 254, 0.02)', border: '1px solid rgba(0, 242, 254, 0.1)', padding: '10px', borderRadius: '6px' }}>
                    {fundamentalRegime.includes('QUALITY') && (
                      <span><strong>💎 Growth Quality Regime active:</strong> Heavy allocation (50%) is automatically mapped to the Temporal Fusion Transformer (TFT) self-attention networks. Since the stock exhibits superior capital efficiency (high ROE/ROCE) and safe leverage levels, we compress the Conformal Risk bands by <strong>5% (0.95x multiplier)</strong> to reflect high forecasting certainty.</span>
                    )}
                    {fundamentalRegime.includes('RISK') && (
                      <span><strong>⚠️ High Leverage/Risk Regime active:</strong> Weights are redirected to Robust Gradient Boosting Trees (40%) to handle non-linear market shocks. Due to high debt ratios or capital weakness, Conformal Uncertainty envelopes are expanded by <strong>25% (1.25x multiplier)</strong> to hedge against sudden credit or solvency shocks.</span>
                    )}
                    {fundamentalRegime.includes('VALUE') && (
                      <span><strong>📈 Value/Cyclical Regime active:</strong> System redirects priority allocation to linear and historical trend models: Robust Ridge (35%) and Holt-Winters (30%). These assets tend to exhibit long-term mean reversion, which linear regressors capture with optimal bias/variance tradeoffs.</span>
                    )}
                    {fundamentalRegime.includes('STANDARD') && (
                      <span><strong>⚖️ Standard Composite Regime active:</strong> Allocations are calibrated using out-of-fold validation MAPE performance (40%) combined with a balanced prior (60%), keeping a standard 1.0x conformal interval width.</span>
                    )}
                  </div>

                  {ensembleWeights && (
                    <div style={{ marginTop: '4px' }}>
                      <span style={{ fontSize: '0.75rem', fontWeight: '700', color: '#fff', display: 'block', marginBottom: '8px' }}>Active Model Weights breakdown:</span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ width: '60px', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>TFT (Attn):</span>
                          <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ width: `${ensembleWeights.tft * 100}%`, background: 'var(--accent-cyan)', height: '100%' }} />
                          </div>
                          <span style={{ fontSize: '0.75rem', fontWeight: '600', width: '35px', textAlign: 'right' }}>{(ensembleWeights.tft * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ width: '60px', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Ridge (Lin):</span>
                          <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ width: `${ensembleWeights.ridge * 100}%`, background: '#ffeb3b', height: '100%' }} />
                          </div>
                          <span style={{ fontSize: '0.75rem', fontWeight: '600', width: '35px', textAlign: 'right' }}>{(ensembleWeights.ridge * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ width: '60px', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>GBR (Tree):</span>
                          <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ width: `${ensembleWeights.gbr * 100}%`, background: '#00e676', height: '100%' }} />
                          </div>
                          <span style={{ fontSize: '0.75rem', fontWeight: '600', width: '35px', textAlign: 'right' }}>{(ensembleWeights.gbr * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ width: '60px', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>HW (Trend):</span>
                          <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ width: `${ensembleWeights.hw * 100}%`, background: '#ff4081', height: '100%' }} />
                          </div>
                          <span style={{ fontSize: '0.75rem', fontWeight: '600', width: '35px', textAlign: 'right' }}>{(ensembleWeights.hw * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
