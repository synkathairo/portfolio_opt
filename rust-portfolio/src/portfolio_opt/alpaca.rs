use apca::api::v2::account;
use apca::api::v2::order::{self, Amount, CreateReqInit, Order, Side, TimeInForce, Type};
use apca::api::v2::orders::{self, ListReq, Status};
use apca::api::v2::positions;
use apca::data::v2::Feed;
use apca::data::v2::bars::{
    self as stock_bars, Adjustment as AlpacaAdjustment, ListReqInit as BarsListReqInit, TimeFrame,
};
use apca::{ApiInfo, Client as AlpacaClient};
use chrono::{Duration, Utc};
use num_decimal::Num;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use yfinance_rs::core::conversions::money_to_f64;
use yfinance_rs::{Interval, Range, Ticker, YfClient};

use crate::portfolio_opt::types::{AccountSnapshot, OrderPlan, Position};

#[derive(Debug, Clone, Copy, Default)]
pub struct YahooCacheOptions {
    pub use_cache: bool,
    pub refresh_cache: bool,
    pub offline: bool,
}

fn num_to_f64(num: Num) -> f64 {
    num.to_string().parse().unwrap_or(0.0)
}

pub struct PortfolioClients {
    alpaca: Option<AlpacaClient>,
    yf: YfClient,
}

impl PortfolioClients {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let api_info = ApiInfo::from_env()?;
        Ok(Self {
            alpaca: Some(AlpacaClient::new(api_info)),
            yf: YfClient::default(),
        })
    }

    pub fn yahoo_only() -> Self {
        Self {
            alpaca: None,
            yf: YfClient::default(),
        }
    }

    pub async fn get_account(&self) -> Result<AccountSnapshot, Box<dyn std::error::Error>> {
        let alpaca = self.alpaca_client()?;
        let info = alpaca.issue::<account::Get>(&()).await?;
        Ok(AccountSnapshot {
            equity: num_to_f64(info.equity),
        })
    }

    pub async fn get_positions(&self) -> Result<Vec<Position>, Box<dyn std::error::Error>> {
        let alpaca = self.alpaca_client()?;
        let positions = alpaca.issue::<positions::List>(&()).await?;
        Ok(positions
            .into_iter()
            .map(|p| Position {
                symbol: p.symbol,
                qty: num_to_f64(p.quantity),
                market_value: p.market_value.map(num_to_f64).unwrap_or(0.0),
            })
            .collect())
    }

    pub async fn get_open_orders(&self) -> Result<Vec<Order>, Box<dyn std::error::Error>> {
        let alpaca = self.alpaca_client()?;
        let req = ListReq {
            status: Status::Open,
            ..Default::default()
        };
        alpaca
            .issue::<orders::List>(&req)
            .await
            .map_err(|e| e.into())
    }

    pub async fn submit_order_plan(
        &self,
        plans: &[OrderPlan],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alpaca = self.alpaca_client()?;
        for plan in plans {
            let side = if plan.side == "buy" {
                Side::Buy
            } else {
                Side::Sell
            };

            let req_init = CreateReqInit {
                type_: Type::Market,
                time_in_force: TimeInForce::Day,
                ..Default::default()
            };

            let notional_val: Num = format!("{}", plan.notional_usd).parse().unwrap();
            let req = req_init.init(
                plan.symbol.as_str(),
                side,
                Amount::Notional {
                    notional: notional_val,
                },
            );

            if let Err(e) = alpaca.issue::<order::Create>(&req).await {
                eprintln!("Failed to submit order for {}: {}", plan.symbol, e);
            }
        }
        Ok(())
    }

    fn alpaca_client(&self) -> Result<&AlpacaClient, Box<dyn std::error::Error>> {
        self.alpaca
            .as_ref()
            .ok_or_else(|| "Alpaca client is not configured for this command".into())
    }

    fn alpaca_feed(&self) -> Feed {
        match std::env::var("APCA_DATA_FEED")
            .unwrap_or_else(|_| "iex".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "sip" => Feed::SIP,
            _ => Feed::IEX,
        }
    }

    fn alpaca_feed_name(&self) -> &'static str {
        match self.alpaca_feed() {
            Feed::SIP => "sip",
            Feed::IEX => "iex",
            _ => "iex",
        }
    }

    #[allow(dead_code)]
    pub async fn fetch_yahoo_closes(
        &self,
        symbols: &[String],
        period_days: usize,
    ) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
        self.fetch_yahoo_closes_with_options(symbols, period_days, YahooCacheOptions::default())
            .await
    }

    pub async fn fetch_yahoo_closes_with_options(
        &self,
        symbols: &[String],
        period_days: usize,
        cache_options: YahooCacheOptions,
    ) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
        if symbols.is_empty() {
            return Ok(HashMap::new());
        }

        if cache_options.use_cache || cache_options.refresh_cache || cache_options.offline {
            return self
                .fetch_yahoo_closes_with_cache(symbols, period_days, cache_options)
                .await;
        }

        let close_maps = self.fetch_yahoo_close_maps(symbols, period_days).await?;
        align_close_maps(symbols, &close_maps)
    }

    pub async fn fetch_alpaca_closes_with_options(
        &self,
        symbols: &[String],
        lookback_days: usize,
        cache_options: YahooCacheOptions,
    ) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
        if symbols.is_empty() {
            return Ok(HashMap::new());
        }

        if cache_options.use_cache || cache_options.refresh_cache || cache_options.offline {
            return self
                .fetch_alpaca_closes_with_cache(symbols, lookback_days, cache_options)
                .await;
        }

        let close_maps = self.fetch_alpaca_close_maps(symbols, lookback_days).await?;
        close_maps_to_trailing_map(symbols, &close_maps, lookback_days)
    }

    async fn fetch_yahoo_closes_with_cache(
        &self,
        symbols: &[String],
        period_days: usize,
        cache_options: YahooCacheOptions,
    ) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
        let mut cached_by_symbol = BTreeMap::new();
        let mut missing_symbols = Vec::new();

        for symbol in symbols {
            match read_symbol_cache(symbol)? {
                Some(closes) if !closes.is_empty() => {
                    cached_by_symbol.insert(symbol.clone(), closes);
                }
                _ => missing_symbols.push(symbol.clone()),
            }
        }

        if cache_options.offline && !missing_symbols.is_empty() {
            return Err(format!(
                "Offline mode requested but yfinance cache is missing for: {}",
                missing_symbols.join(", ")
            )
            .into());
        }

        let symbols_to_fetch = if cache_options.refresh_cache {
            symbols.to_vec()
        } else {
            missing_symbols
        };
        let fetched = self
            .fetch_yahoo_close_maps(&symbols_to_fetch, period_days)
            .await?;
        for (symbol, closes) in fetched {
            if closes.is_empty() {
                continue;
            }
            write_symbol_cache(&symbol, &closes)?;
            cached_by_symbol.insert(symbol, closes);
        }

        if symbols
            .iter()
            .any(|symbol| !cached_by_symbol.contains_key(symbol))
        {
            let still_missing = symbols
                .iter()
                .filter(|symbol| !cached_by_symbol.contains_key(*symbol))
                .cloned()
                .collect::<Vec<_>>();
            return Err(format!(
                "Failed to load yfinance history for: {}",
                still_missing.join(", ")
            )
            .into());
        }

        align_close_maps(symbols, &cached_by_symbol)
    }

    async fn fetch_alpaca_closes_with_cache(
        &self,
        symbols: &[String],
        lookback_days: usize,
        cache_options: YahooCacheOptions,
    ) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
        let feed_name = self.alpaca_feed_name();
        let mut cached_by_symbol = BTreeMap::new();
        let mut missing_or_short_symbols = Vec::new();

        for symbol in symbols {
            match read_alpaca_symbol_cache(symbol, feed_name)? {
                Some(closes) if closes.len() >= lookback_days => {
                    cached_by_symbol.insert(symbol.clone(), closes);
                }
                Some(closes) => {
                    cached_by_symbol.insert(symbol.clone(), closes);
                    missing_or_short_symbols.push(symbol.clone());
                }
                None => missing_or_short_symbols.push(symbol.clone()),
            }
        }

        if cache_options.offline && !missing_or_short_symbols.is_empty() {
            return Err(format!(
                "Offline mode requested but Alpaca daily close cache is missing or short for: {}",
                missing_or_short_symbols.join(", ")
            )
            .into());
        }

        let symbols_to_fetch = if cache_options.refresh_cache {
            symbols.to_vec()
        } else {
            missing_or_short_symbols
        };
        let fetched = self
            .fetch_alpaca_close_maps(&symbols_to_fetch, lookback_days)
            .await?;
        for (symbol, closes) in fetched {
            if closes.is_empty() {
                continue;
            }
            let mut merged = cached_by_symbol.remove(&symbol).unwrap_or_default();
            merged.extend(closes);
            let trimmed = trim_close_map(&merged, lookback_days);
            write_alpaca_symbol_cache(&symbol, feed_name, &trimmed)?;
            cached_by_symbol.insert(symbol, trimmed);
        }

        close_maps_to_trailing_map(symbols, &cached_by_symbol, lookback_days)
    }

    async fn fetch_yahoo_close_maps(
        &self,
        symbols: &[String],
        period_days: usize,
    ) -> Result<BTreeMap<String, BTreeMap<String, f64>>, Box<dyn std::error::Error>> {
        let mut closes = BTreeMap::new();
        let range = if period_days > 600 {
            Range::Max
        } else if period_days > 252 {
            Range::Y2
        } else {
            Range::Y1
        };

        for symbol in symbols {
            let ticker = Ticker::new(&self.yf, symbol);
            if let Ok(history) = ticker.history(Some(range), Some(Interval::D1), false).await {
                let close_map = history
                    .iter()
                    .map(|candle| {
                        (
                            candle.ts.format("%Y-%m-%d").to_string(),
                            money_to_f64(&candle.close),
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                if !close_map.is_empty() {
                    closes.insert(symbol.clone(), close_map);
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Ok(closes)
    }

    async fn fetch_alpaca_close_maps(
        &self,
        symbols: &[String],
        lookback_days: usize,
    ) -> Result<BTreeMap<String, BTreeMap<String, f64>>, Box<dyn std::error::Error>> {
        let alpaca = self.alpaca_client()?;
        let end = Utc::now();
        let start = end - Duration::days((lookback_days * 3).max(30) as i64);
        let feed = self.alpaca_feed();
        let mut closes = BTreeMap::new();

        for symbol in symbols {
            let req = BarsListReqInit {
                limit: Some(lookback_days + 5),
                adjustment: Some(AlpacaAdjustment::All),
                feed: Some(feed),
                ..Default::default()
            }
            .init(symbol.clone(), start, end, TimeFrame::OneDay);
            let bars = alpaca.issue::<stock_bars::List>(&req).await?;
            let symbol_closes = bars
                .bars
                .iter()
                .map(|bar| {
                    (
                        bar.time.format("%Y-%m-%d").to_string(),
                        num_to_f64(bar.close.clone()),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            if !symbol_closes.is_empty() {
                closes.insert(
                    symbol.clone(),
                    trim_close_map(&symbol_closes, lookback_days),
                );
            }
        }

        Ok(closes)
    }
}

fn align_close_maps(
    symbols: &[String],
    closes_by_symbol: &BTreeMap<String, BTreeMap<String, f64>>,
) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    let mut common_dates: Option<BTreeSet<String>> = None;
    for symbol in symbols {
        let closes = closes_by_symbol
            .get(symbol)
            .ok_or_else(|| format!("Missing data for {symbol}"))?;
        let dates = closes.keys().cloned().collect::<BTreeSet<_>>();
        common_dates = Some(match common_dates {
            Some(existing) => existing.intersection(&dates).cloned().collect(),
            None => dates,
        });
    }

    let dates = common_dates.unwrap_or_default();
    if dates.len() < 2 {
        return Err("Not enough date-aligned common history.".into());
    }

    let mut aligned = HashMap::new();
    for symbol in symbols {
        let closes = closes_by_symbol
            .get(symbol)
            .ok_or_else(|| format!("Missing data for {symbol}"))?;
        let values = dates
            .iter()
            .filter_map(|date| closes.get(date).copied())
            .collect::<Vec<_>>();
        aligned.insert(symbol.clone(), values);
    }
    Ok(aligned)
}

fn close_maps_to_trailing_map(
    symbols: &[String],
    closes_by_symbol: &BTreeMap<String, BTreeMap<String, f64>>,
    lookback_days: usize,
) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    let mut result = HashMap::new();
    for symbol in symbols {
        let closes = closes_by_symbol
            .get(symbol)
            .ok_or_else(|| format!("Missing data for {symbol}"))?;
        if closes.len() < lookback_days {
            return Err(format!(
                "Not enough Alpaca daily bars returned for {symbol}. Requested {lookback_days} trading days, got {}.",
                closes.len()
            )
            .into());
        }
        let values = closes
            .values()
            .rev()
            .take(lookback_days)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        result.insert(symbol.clone(), values);
    }
    Ok(result)
}

fn trim_close_map(closes: &BTreeMap<String, f64>, lookback_days: usize) -> BTreeMap<String, f64> {
    closes
        .iter()
        .rev()
        .take(lookback_days)
        .map(|(date, close)| (date.clone(), *close))
        .collect::<BTreeMap<_, _>>()
}

fn read_symbol_cache(
    symbol: &str,
) -> Result<Option<BTreeMap<String, f64>>, Box<dyn std::error::Error>> {
    let path = symbol_closes_cache_path(symbol);
    if !path.exists() {
        return Ok(None);
    }
    let payload: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    if let Some(closes) = payload.get("closes").and_then(|value| value.as_object()) {
        let mut result = BTreeMap::new();
        for (date, close) in closes {
            if let Some(value) = close.as_f64() {
                result.insert(date[..date.len().min(10)].to_string(), value);
            }
        }
        return Ok(Some(result));
    }
    if let Some(rows) = payload.as_array() {
        let mut result = BTreeMap::new();
        for row in rows {
            let Some(timestamp) = row.get("timestamp").and_then(|value| value.as_str()) else {
                continue;
            };
            let Some(close) = row.get("close").and_then(|value| value.as_f64()) else {
                continue;
            };
            result.insert(timestamp[..timestamp.len().min(10)].to_string(), close);
        }
        return Ok(Some(result));
    }
    Ok(None)
}

fn write_symbol_cache(
    symbol: &str,
    closes: &BTreeMap<String, f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = symbol_closes_cache_path(symbol);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::json!({
        "symbol": symbol,
        "source": "yfinance",
        "adjustment": "auto",
        "closes": closes,
    });
    fs::write(path, serde_json::to_string_pretty(&payload)?)?;
    Ok(())
}

fn symbol_closes_cache_path(symbol: &str) -> PathBuf {
    let safe_symbol = symbol
        .to_uppercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let payload = format!(
        "{{\"adjustment\": \"auto\", \"kind\": \"yfinance_closes_v2\", \"symbol\": {}}}",
        serde_json::to_string(symbol).expect("symbol serialization cannot fail")
    );
    let digest = Sha256::digest(payload.as_bytes());
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    PathBuf::from(".cache").join(format!(
        "yfinance_closes_v2_{}_{}.json",
        safe_symbol,
        &hex[..16]
    ))
}

fn read_alpaca_symbol_cache(
    symbol: &str,
    feed: &str,
) -> Result<Option<BTreeMap<String, f64>>, Box<dyn std::error::Error>> {
    let path = alpaca_symbol_closes_cache_path(symbol, feed);
    if !path.exists() {
        return Ok(None);
    }
    read_close_payload(&path)
}

fn write_alpaca_symbol_cache(
    symbol: &str,
    feed: &str,
    closes: &BTreeMap<String, f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = alpaca_symbol_closes_cache_path(symbol, feed);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::json!({
        "symbol": symbol,
        "source": "alpaca",
        "feed": feed,
        "adjustment": "all",
        "closes": closes,
    });
    fs::write(path, serde_json::to_string_pretty(&payload)?)?;
    Ok(())
}

fn alpaca_symbol_closes_cache_path(symbol: &str, feed: &str) -> PathBuf {
    let safe_symbol = symbol
        .to_uppercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let payload = format!(
        "{{\"adjustment\": \"all\", \"feed\": {}, \"kind\": \"daily_closes_v2\", \"symbol\": {}}}",
        serde_json::to_string(feed).expect("feed serialization cannot fail"),
        serde_json::to_string(symbol).expect("symbol serialization cannot fail")
    );
    let digest = Sha256::digest(payload.as_bytes());
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    PathBuf::from(".cache").join(format!(
        "daily_closes_v2_{}_{}.json",
        safe_symbol,
        &hex[..16]
    ))
}

fn read_close_payload(
    path: &PathBuf,
) -> Result<Option<BTreeMap<String, f64>>, Box<dyn std::error::Error>> {
    let payload: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    if let Some(closes) = payload.get("closes").and_then(|value| value.as_object()) {
        let mut result = BTreeMap::new();
        for (date, close) in closes {
            if let Some(value) = close.as_f64() {
                result.insert(date[..date.len().min(10)].to_string(), value);
            }
        }
        return Ok(Some(result));
    }
    Ok(None)
}
