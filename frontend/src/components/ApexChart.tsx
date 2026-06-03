'use client';

import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ColorType, LineStyle } from 'lightweight-charts';

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
  tft_close?: number;
  ridge_close?: number;
  gbr_close?: number;
  hw_close?: number;
  lower_90: number;
  upper_90: number;
  lower_95: number;
  upper_95: number;
}

interface ApexChartProps {
  data: DataPoint[];
  forecasts?: ForecastPoint[];
  activeTab: 'forecast' | 'emd' | 'ffd' | 'hmm' | 'rsi' | 'macd';
}

export default function ApexChart({ data, forecasts = [], activeTab }: ApexChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    // Clear container
    chartContainerRef.current.innerHTML = '';

    // Initialize Chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0b0f19' },
        textColor: '#8e9cae',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.08)',
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.08)',
      },
      width: chartContainerRef.current.clientWidth,
      height: 480,
    });

    chartRef.current = chart;

    if (activeTab === 'forecast') {
      // 1. Raw Close Price Series
      const rawLineSeries = chart.addLineSeries({
        color: '#4facfe',
        lineWidth: 2,
        title: 'Actual Price',
      });
      rawLineSeries.setData(data.map(d => ({ time: d.timestamp, value: d.close_raw })));

      // 2. Forecast Dash Overlay
      if (forecasts.length > 0) {
        // Thick glow line for weighted ensemble forecast
        const forecastSeries = chart.addLineSeries({
          color: '#00f2fe',
          lineWidth: 3,
          title: 'Weighted Ensemble Forecast',
        });

        // Individual TFT Forecast Line
        const tftSeries = chart.addLineSeries({
          color: '#e040fb',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          title: 'Self-Attention (TFT) Forecast',
        });

        // Individual Ridge Forecast Line
        const ridgeSeries = chart.addLineSeries({
          color: '#ffeb3b',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          title: 'Ridge Regression Forecast',
        });

        // Individual GBR Forecast Line
        const gbrSeries = chart.addLineSeries({
          color: '#00e676',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          title: 'Gradient Boosting Forecast',
        });

        // Individual Holt-Winters Forecast Line
        const hwSeries = chart.addLineSeries({
          color: '#ff4081',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          title: 'Holt-Winters Forecast',
        });

        // 90% Bounds (Green Dotted)
        const upper90Series = chart.addLineSeries({
          color: 'rgba(0, 230, 118, 0.5)',
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          title: '90% Upper Bound',
        });
        const lower90Series = chart.addLineSeries({
          color: 'rgba(0, 230, 118, 0.5)',
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          title: '90% Lower Bound',
        });

        // 95% Bounds (Cyan Solid Transparent)
        const upper95Series = chart.addLineSeries({
          color: 'rgba(0, 242, 254, 0.3)',
          lineWidth: 1,
          title: '95% Upper Bound',
        });
        const lower95Series = chart.addLineSeries({
          color: 'rgba(0, 242, 254, 0.3)',
          lineWidth: 1,
          title: '95% Lower Bound',
        });

        // Concat transitions
        const lastHist = data[data.length - 1];
        
        forecastSeries.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.forecast_close }))
        ]);

        tftSeries.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.tft_close ?? f.forecast_close }))
        ]);

        ridgeSeries.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.ridge_close ?? f.forecast_close }))
        ]);

        gbrSeries.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.gbr_close ?? f.forecast_close }))
        ]);

        hwSeries.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.hw_close ?? f.forecast_close }))
        ]);

        upper90Series.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.upper_90 }))
        ]);
        lower90Series.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.lower_90 }))
        ]);

        upper95Series.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.upper_95 }))
        ]);
        lower95Series.setData([
          { time: lastHist.timestamp, value: lastHist.close_raw },
          ...forecasts.map(f => ({ time: f.timestamp, value: f.lower_95 }))
        ]);
      }
    } 
    else if (activeTab === 'emd') {
      // Raw price
      const rawLineSeries = chart.addLineSeries({
        color: 'rgba(79, 172, 254, 0.4)',
        lineWidth: 1,
        title: 'Raw Price',
      });
      rawLineSeries.setData(data.map(d => ({ time: d.timestamp, value: d.close_raw })));

      // EMD Smoothed cycle wave
      const emdLineSeries = chart.addLineSeries({
        color: '#00f2fe',
        lineWidth: 3,
        title: 'EMD Smoothed Cycles',
      });
      emdLineSeries.setData(data.map(d => ({ time: d.timestamp, value: d.close_emd_smoothed })));
    } 
    else if (activeTab === 'ffd') {
      // FFD stationary series
      const ffdLineSeries = chart.addLineSeries({
        color: '#00e676',
        lineWidth: 2,
        title: 'Stationary FFD (d)',
      });
      ffdLineSeries.setData(data.map(d => ({ time: d.timestamp, value: d.close_ffd })));

      // Horizontal zero baseline to demonstrate mean-reverting stationarity
      const zeroLine = chart.addLineSeries({
        color: 'rgba(255, 255, 255, 0.15)',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Mean Target',
      });
      zeroLine.setData(data.map(d => ({ time: d.timestamp, value: 0.0 })));
    } 
    else if (activeTab === 'hmm') {
      // Raw close price
      const rawLineSeries = chart.addLineSeries({
        color: '#4facfe',
        lineWidth: 2,
        title: 'Raw Close Price',
      });
      rawLineSeries.setData(data.map(d => ({ time: d.timestamp, value: d.close_raw })));

      // Render regime transition markers along the price series
      const markers = data
        .map((d, i) => {
          const isHighVol = d.regime === 1;
          const prevIsHighVol = i > 0 ? data[i - 1].regime === 1 : false;

          if (isHighVol && !prevIsHighVol) {
            return {
              time: d.timestamp,
              position: 'aboveBar' as const,
              color: '#ff1744',
              shape: 'arrowDown' as const,
              text: 'BEAR SHIFT (HIGH VOL)',
            };
          } else if (!isHighVol && prevIsHighVol) {
            return {
              time: d.timestamp,
              position: 'belowBar' as const,
              color: '#00e676',
              shape: 'arrowUp' as const,
              text: 'BULL SHIFT (LOW VOL)',
            };
          }
          return null;
        })
        .filter(Boolean) as any[];

      rawLineSeries.setMarkers(markers);
    }
    else if (activeTab === 'rsi') {
      // Relative Strength Index line
      const rsiSeries = chart.addLineSeries({
        color: '#e040fb',
        lineWidth: 2,
        title: 'Relative Strength Index (RSI)',
      });
      rsiSeries.setData(data.map(d => ({ time: d.timestamp, value: d.rsi ?? 50.0 })));

      // Overbought threshold at 70
      const overboughtLine = chart.addLineSeries({
        color: '#ff1744',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Overbought (70)',
      });
      overboughtLine.setData(data.map(d => ({ time: d.timestamp, value: 70.0 })));

      // Oversold threshold at 30
      const oversoldLine = chart.addLineSeries({
        color: '#00e676',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Oversold (30)',
      });
      oversoldLine.setData(data.map(d => ({ time: d.timestamp, value: 30.0 })));

      // Neutral reference baseline at 50
      const neutralLine = chart.addLineSeries({
        color: 'rgba(255, 255, 255, 0.15)',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Neutral (50)',
      });
      neutralLine.setData(data.map(d => ({ time: d.timestamp, value: 50.0 })));
    }
    else if (activeTab === 'macd') {
      // Moving Average Convergence Divergence line
      const macdSeries = chart.addLineSeries({
        color: '#00f2fe',
        lineWidth: 2,
        title: 'MACD Indicator',
      });
      macdSeries.setData(data.map(d => ({ time: d.timestamp, value: d.macd ?? 0.0 })));

      // Neutral zero line baseline
      const zeroLine = chart.addLineSeries({
        color: 'rgba(255, 255, 255, 0.15)',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Zero Line',
      });
      zeroLine.setData(data.map(d => ({ time: d.timestamp, value: 0.0 })));
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, forecasts, activeTab]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div 
        ref={chartContainerRef} 
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
