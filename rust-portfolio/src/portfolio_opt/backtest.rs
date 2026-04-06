use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct BacktestResult {
    pub final_value: f64,
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub max_drawdown: f64,
    pub rebalance_count: usize,
    pub average_turnover: f64,
    pub daily_values: Vec<f64>,
}

pub fn run_dual_momentum_backtest(
    symbols: &[String],
    closes_by_symbol: &HashMap<String, Vec<f64>>,
    asset_classes: &HashMap<String, String>,
    lookback_days: usize,
    rebalance_every: usize,
    top_k: usize,
    absolute_threshold: f64,
) -> Result<BacktestResult, Box<dyn std::error::Error>> {
    // Build price matrix (symbols x days) from the closes data
    let mut price_data = Vec::new();
    let mut num_days = 0;

    for symbol in symbols {
        let closes = closes_by_symbol.get(symbol)
            .ok_or_else(|| format!("Missing data for {}", symbol))?;
        
        if num_days == 0 {
            num_days = closes.len();
        } else if closes.len() != num_days {
            return Err(format!("Inconsistent history length for {}", symbol).into());
        }
        
        price_data.extend_from_slice(closes);
    }

    let price_matrix = Array2::from_shape_vec((symbols.len(), num_days), price_data)
        .expect("Failed to reshape price data into matrix");

    let returns = (&price_matrix.slice(ndarray::s![.., 1..]) / &price_matrix.slice(ndarray::s![.., ..-1])) - 1.0;
    let mut portfolio_value = 1.0;
    let mut weights = Array1::<f64>::zeros(symbols.len());
    let mut daily_values = vec![1.0];
    let mut rebalance_count = 0;
    let mut turnovers = Vec::new();

    let risky_symbols: Vec<usize> = symbols.iter().enumerate().filter(|(_, s)| {
        let class = asset_classes.get(s.as_str()).map(|s| s.as_str()).unwrap_or("");
        !class.starts_with("bond") && class != "cash_like"
    }).map(|(i, _)| i).collect();

    let defensive_symbols: Vec<usize> = symbols.iter().enumerate().filter(|(_, s)| {
        let class = asset_classes.get(s.as_str()).map(|s| s.as_str()).unwrap_or("");
        class.starts_with("bond") || class == "cash_like"
    }).map(|(i, _)| i).collect();

    let cash_like_index = symbols.iter().position(|s| asset_classes.get(s.as_str()).map(|s| s.as_str()) == Some("cash_like"));

    for step in lookback_days..returns.ncols() {
        let mut target_weights = Array1::<f64>::zeros(symbols.len());

        if (step - lookback_days) % rebalance_every == 0 {
            let trailing_returns: Vec<(usize, f64)> = risky_symbols.iter().filter_map(|&idx| {
                let ret = price_matrix[[idx, step]] / price_matrix[[idx, step - lookback_days]] - 1.0;
                let floor = cash_like_index.map(|i| price_matrix[[i, step]] / price_matrix[[i, step - lookback_days]] - 1.0).unwrap_or(absolute_threshold);
                if ret > absolute_threshold.max(floor) {
                    Some((idx, ret))
                } else {
                    None
                }
            }).collect();

            let mut ranked = trailing_returns;
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let selected: Vec<_> = ranked.into_iter().take(top_k).collect();

            if selected.is_empty() {
                if !defensive_symbols.is_empty() {
                    let w = 1.0 / defensive_symbols.len() as f64;
                    for &idx in &defensive_symbols {
                        target_weights[idx] = w;
                    }
                }
            } else {
                let w = 1.0 / selected.len() as f64;
                for &(idx, _) in &selected {
                    target_weights[idx] = w;
                }
            }

            turnovers.push((target_weights.clone() - weights.clone()).mapv(f64::abs).sum());
            weights = target_weights;
            rebalance_count += 1;
        }

        let period_return = weights.dot(&returns.column(step));
        portfolio_value *= 1.0 + period_return;
        daily_values.push(portfolio_value);
    }

    let total_return = portfolio_value - 1.0;
    let n_days = (returns.ncols() - lookback_days) as f64;
    let annualized_return = portfolio_value.powf(252.0 / n_days) - 1.0;
    let vols: Vec<f64> = returns.axis_iter(ndarray::Axis(0)).map(|row| row.std(0.0)).collect();
    let avg_vol = vols.iter().sum::<f64>() / vols.len() as f64;
    let annualized_vol = avg_vol * (252.0_f64).sqrt();
    let max_dd = calculate_max_drawdown(&daily_values);

    Ok(BacktestResult {
        final_value: portfolio_value,
        total_return,
        annualized_return,
        annualized_volatility: annualized_vol,
        max_drawdown: max_dd,
        rebalance_count,
        average_turnover: if !turnovers.is_empty() { turnovers.iter().sum::<f64>() / turnovers.len() as f64 } else { 0.0 },
        daily_values,
    })
}

fn calculate_max_drawdown(values: &[f64]) -> f64 {
    let mut peak = values[0];
    let mut max_dd = 0.0;
    for &v in values {
        if v > peak { peak = v; }
        let dd = (peak - v) / peak;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

pub fn calculate_benchmark_stats(closes: &[f64]) -> serde_json::Value {
    if closes.len() < 2 {
        return serde_json::json!({});
    }
    let start_price = closes[0];
    let end_price = closes[closes.len() - 1];
    let total_return = (end_price / start_price) - 1.0;
    let n_days = closes.len() - 1;
    let ann_return = (end_price / start_price).powf(252.0 / n_days as f64) - 1.0;

    // Volatility
    let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let daily_vol = variance.sqrt();
    let ann_vol = daily_vol * (252.0_f64).sqrt();

    // Max Drawdown
    let mut peak = start_price;
    let mut max_dd = 0.0;
    for &p in closes.iter() {
        if p > peak { peak = p; }
        let dd = (peak - p) / peak;
        if dd > max_dd { max_dd = dd; }
    }

    serde_json::json!({
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd
    })
}
