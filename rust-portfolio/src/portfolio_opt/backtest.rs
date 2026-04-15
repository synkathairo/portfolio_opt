use ndarray::{Array1, Array2};
use std::collections::HashMap;

const TRADING_DAYS_PER_YEAR: f64 = 252.0;

#[derive(Debug, Clone)]
pub struct DualMomentumOptions {
    pub lookback_days: usize,
    pub top_k: usize,
    pub absolute_threshold: f64,
    pub weighting: String,
    pub softmax_temperature: f64,
    pub target_vol: Option<f64>,
    pub max_single_weight: Option<f64>,
    pub vol_window: usize,
    pub trailing_stop: Option<f64>,
}

impl Default for DualMomentumOptions {
    fn default() -> Self {
        Self {
            lookback_days: 252,
            top_k: 2,
            absolute_threshold: 0.0,
            weighting: "equal".into(),
            softmax_temperature: 0.05,
            target_vol: None,
            max_single_weight: None,
            vol_window: 63,
            trailing_stop: None,
        }
    }
}

pub struct BacktestResult {
    pub final_value: f64,
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub max_drawdown: f64,
    pub rebalance_count: usize,
    pub average_turnover: f64,
    pub latest_weights: Vec<f64>,
    pub daily_values: Vec<f64>,
}

#[allow(dead_code)]
pub fn compute_dual_momentum_targets(
    symbols: &[String],
    closes_by_symbol: &HashMap<String, Vec<f64>>,
    asset_classes: &HashMap<String, String>,
    lookback_days: usize,
    top_k: usize,
    absolute_threshold: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let options = DualMomentumOptions {
        lookback_days,
        top_k,
        absolute_threshold,
        ..Default::default()
    };
    compute_dual_momentum_targets_with_options(symbols, closes_by_symbol, asset_classes, &options)
}

pub fn compute_dual_momentum_targets_with_options(
    symbols: &[String],
    closes_by_symbol: &HashMap<String, Vec<f64>>,
    asset_classes: &HashMap<String, String>,
    options: &DualMomentumOptions,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let price_matrix = build_price_matrix(symbols, closes_by_symbol)?;
    if price_matrix.ncols() < options.lookback_days + 1 {
        return Err(format!(
            "Not enough history: have {}, need {}",
            price_matrix.ncols(),
            options.lookback_days + 1
        )
        .into());
    }

    let returns = (&price_matrix.slice(ndarray::s![.., 1..])
        / &price_matrix.slice(ndarray::s![.., ..-1]))
        - 1.0;
    let step = returns.ncols() - 1;
    compute_dual_momentum_weights_at_step(
        symbols,
        asset_classes,
        &price_matrix,
        &returns,
        step,
        options,
    )
}

#[allow(dead_code)]
pub fn run_dual_momentum_backtest(
    symbols: &[String],
    closes_by_symbol: &HashMap<String, Vec<f64>>,
    asset_classes: &HashMap<String, String>,
    lookback_days: usize,
    rebalance_every: usize,
    top_k: usize,
    absolute_threshold: f64,
) -> Result<BacktestResult, Box<dyn std::error::Error>> {
    let options = DualMomentumOptions {
        lookback_days,
        top_k,
        absolute_threshold,
        ..Default::default()
    };
    run_dual_momentum_backtest_with_options(
        symbols,
        closes_by_symbol,
        asset_classes,
        rebalance_every,
        &options,
    )
}

pub fn run_dual_momentum_backtest_with_options(
    symbols: &[String],
    closes_by_symbol: &HashMap<String, Vec<f64>>,
    asset_classes: &HashMap<String, String>,
    rebalance_every: usize,
    options: &DualMomentumOptions,
) -> Result<BacktestResult, Box<dyn std::error::Error>> {
    if rebalance_every == 0 {
        return Err("rebalance_every must be greater than zero".into());
    }

    let price_matrix = build_price_matrix(symbols, closes_by_symbol)?;
    if price_matrix.ncols() < options.lookback_days + 1 {
        return Err("Not enough price history to run the backtest.".into());
    }

    let returns = (&price_matrix.slice(ndarray::s![.., 1..])
        / &price_matrix.slice(ndarray::s![.., ..-1]))
        - 1.0;
    let mut portfolio_value = 1.0;
    let mut weights = Array1::<f64>::zeros(symbols.len());
    let mut portfolio_returns = Vec::new();
    let mut daily_values = vec![1.0];
    let mut rebalance_count = 0;
    let mut turnovers = Vec::new();
    let mut asset_peak_price: Option<Array1<f64>> = None;

    for step in options.lookback_days..returns.ncols() {
        if (step - options.lookback_days) % rebalance_every == 0 {
            let target_weights = Array1::from_vec(compute_dual_momentum_weights_at_step(
                symbols,
                asset_classes,
                &price_matrix,
                &returns,
                step,
                options,
            )?);
            turnovers.push((&target_weights - &weights).mapv(f64::abs).sum());
            weights = target_weights;
            rebalance_count += 1;
        }

        if let Some(trailing_stop) = options.trailing_stop {
            let peaks = asset_peak_price
                .get_or_insert_with(|| price_matrix.column(step.saturating_sub(1)).to_owned());
            for idx in 0..symbols.len() {
                if weights[idx] > 0.0 {
                    peaks[idx] = peaks[idx].max(price_matrix[[idx, step]]);
                }
            }
            for idx in 0..symbols.len() {
                if weights[idx] > 0.0 && peaks[idx] > 0.0 {
                    let drawdown_from_peak = (peaks[idx] - price_matrix[[idx, step]]) / peaks[idx];
                    if drawdown_from_peak > trailing_stop {
                        weights[idx] = 0.0;
                        peaks[idx] = 0.0;
                    }
                }
            }
        }

        let period_return = weights.dot(&returns.column(step));
        portfolio_returns.push(period_return);
        portfolio_value *= 1.0 + period_return;
        daily_values.push(portfolio_value);
    }

    let total_return = portfolio_value - 1.0;
    let n_days = portfolio_returns.len().max(1) as f64;
    let annualized_return = portfolio_value.powf(TRADING_DAYS_PER_YEAR / n_days) - 1.0;
    let annualized_vol = std_dev(&portfolio_returns) * TRADING_DAYS_PER_YEAR.sqrt();
    let max_dd = calculate_max_drawdown(&daily_values);

    Ok(BacktestResult {
        final_value: portfolio_value,
        total_return,
        annualized_return,
        annualized_volatility: annualized_vol,
        max_drawdown: max_dd,
        rebalance_count,
        average_turnover: mean_or_zero(&turnovers),
        latest_weights: weights.to_vec(),
        daily_values,
    })
}

fn build_price_matrix(
    symbols: &[String],
    closes_by_symbol: &HashMap<String, Vec<f64>>,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let common_len = symbols
        .iter()
        .map(|symbol| {
            closes_by_symbol
                .get(symbol)
                .map(|closes| closes.len())
                .ok_or_else(|| format!("Missing data for {}", symbol))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .unwrap_or(0);

    if common_len < 2 {
        return Err("Not enough aligned price history.".into());
    }

    let mut price_data = Vec::with_capacity(symbols.len() * common_len);
    for symbol in symbols {
        let closes = closes_by_symbol
            .get(symbol)
            .ok_or_else(|| format!("Missing data for {}", symbol))?;
        price_data.extend_from_slice(&closes[closes.len() - common_len..]);
    }

    Ok(Array2::from_shape_vec(
        (symbols.len(), common_len),
        price_data,
    )?)
}

fn compute_dual_momentum_weights_at_step(
    symbols: &[String],
    asset_classes: &HashMap<String, String>,
    price_matrix: &Array2<f64>,
    returns: &Array2<f64>,
    step: usize,
    options: &DualMomentumOptions,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let risky_indices = risky_indices(symbols, asset_classes);
    if risky_indices.is_empty() {
        return Err("Dual momentum requires at least one risky symbol.".into());
    }
    let defensive_indices = defensive_indices(symbols, asset_classes);
    let cash_like_index = symbols
        .iter()
        .position(|s| asset_classes.get(s.as_str()).map(|s| s.as_str()) == Some("cash_like"));

    let trailing_returns = trailing_returns(price_matrix, step, options.lookback_days);
    let floor = cash_like_index
        .map(|idx| trailing_returns[idx])
        .unwrap_or(options.absolute_threshold);
    let mut ranked = risky_indices
        .iter()
        .filter_map(|&idx| {
            let ret = trailing_returns[idx];
            if ret > options.absolute_threshold.max(floor) {
                Some((idx, ret))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut target_weights = Array1::<f64>::zeros(symbols.len());
    if ranked.is_empty() {
        if !defensive_indices.is_empty() {
            let weight = 1.0 / defensive_indices.len() as f64;
            for idx in defensive_indices {
                target_weights[idx] = weight;
            }
        }
        return Ok(target_weights.to_vec());
    }

    let selected = ranked
        .into_iter()
        .take(options.top_k)
        .collect::<Vec<(usize, f64)>>();
    let trailing_volatility = trailing_volatility(returns, step, options.lookback_days);
    for (idx, weight) in selected_weights(
        &selected,
        &trailing_returns,
        &trailing_volatility,
        &options.weighting,
        options.softmax_temperature,
    )? {
        target_weights[idx] = weight;
    }

    if let Some(max_single_weight) = options.max_single_weight {
        apply_single_weight_cap(&mut target_weights, max_single_weight);
    }
    if let Some(target_vol) = options.target_vol {
        apply_target_volatility(
            &mut target_weights,
            returns,
            step,
            options.vol_window,
            target_vol,
        );
    }

    Ok(target_weights.to_vec())
}

fn risky_indices(symbols: &[String], asset_classes: &HashMap<String, String>) -> Vec<usize> {
    symbols
        .iter()
        .enumerate()
        .filter(|(_, symbol)| {
            let class = asset_classes
                .get(symbol.as_str())
                .map(|s| s.as_str())
                .unwrap_or("");
            !class.starts_with("bond") && class != "cash_like"
        })
        .map(|(idx, _)| idx)
        .collect()
}

fn defensive_indices(symbols: &[String], asset_classes: &HashMap<String, String>) -> Vec<usize> {
    symbols
        .iter()
        .enumerate()
        .filter(|(_, symbol)| {
            let class = asset_classes
                .get(symbol.as_str())
                .map(|s| s.as_str())
                .unwrap_or("");
            class.starts_with("bond") || class == "cash_like"
        })
        .map(|(idx, _)| idx)
        .collect()
}

fn trailing_returns(price_matrix: &Array2<f64>, step: usize, lookback_days: usize) -> Vec<f64> {
    (0..price_matrix.nrows())
        .map(|idx| price_matrix[[idx, step]] / price_matrix[[idx, step - lookback_days]] - 1.0)
        .collect()
}

fn trailing_volatility(returns: &Array2<f64>, step: usize, lookback_days: usize) -> Vec<f64> {
    let start = step.saturating_sub(lookback_days);
    (0..returns.nrows())
        .map(|idx| {
            let values = returns
                .slice(ndarray::s![idx, start..step])
                .iter()
                .copied()
                .collect::<Vec<_>>();
            std_dev(&values)
        })
        .collect()
}

fn selected_weights(
    selected: &[(usize, f64)],
    trailing_returns: &[f64],
    trailing_volatility: &[f64],
    weighting: &str,
    softmax_temperature: f64,
) -> Result<Vec<(usize, f64)>, Box<dyn std::error::Error>> {
    if selected.is_empty() {
        return Ok(Vec::new());
    }

    let raw_weights = match weighting {
        "equal" => vec![1.0; selected.len()],
        "score" => selected
            .iter()
            .map(|(_, score)| score.max(0.0))
            .collect::<Vec<_>>(),
        "inverse-vol" => selected
            .iter()
            .map(|(idx, _)| 1.0 / trailing_volatility[*idx].max(1e-8))
            .collect::<Vec<_>>(),
        "softmax" => {
            let temp = softmax_temperature.max(1e-6);
            let scaled = selected
                .iter()
                .map(|(idx, _)| trailing_returns[*idx] / temp)
                .collect::<Vec<_>>();
            let max_scaled = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            scaled
                .iter()
                .map(|value| (value - max_scaled).exp())
                .collect::<Vec<_>>()
        }
        _ => return Err(format!("Unknown dual momentum weighting mode: {weighting}").into()),
    };

    let normalized = normalize_positive(&raw_weights);
    Ok(selected
        .iter()
        .zip(normalized.iter())
        .map(|((idx, _), weight)| (*idx, *weight))
        .collect())
}

fn normalize_positive(values: &[f64]) -> Vec<f64> {
    let total = values.iter().sum::<f64>();
    if total <= 0.0 {
        return vec![1.0 / values.len() as f64; values.len()];
    }
    values.iter().map(|value| value / total).collect()
}

fn apply_single_weight_cap(weights: &mut Array1<f64>, max_single_weight: f64) {
    let mut excess = 0.0;
    for weight in weights.iter_mut() {
        if *weight > max_single_weight {
            excess += *weight - max_single_weight;
            *weight = max_single_weight;
        } else if *weight < 0.0 {
            excess += weight.abs();
            *weight = 0.0;
        }
    }

    let remaining = weights.iter().filter(|weight| **weight > 0.0).sum::<f64>();
    if remaining > 0.0 {
        for weight in weights.iter_mut() {
            if *weight > 0.0 && *weight < max_single_weight {
                *weight += excess * (*weight / remaining);
            }
        }
    }
}

fn apply_target_volatility(
    weights: &mut Array1<f64>,
    returns: &Array2<f64>,
    step: usize,
    vol_window: usize,
    target_vol: f64,
) {
    let active_indices = weights
        .iter()
        .enumerate()
        .filter_map(|(idx, weight)| if *weight > 0.0 { Some(idx) } else { None })
        .collect::<Vec<_>>();
    if active_indices.is_empty() {
        return;
    }

    let start = step.saturating_sub(vol_window);
    let portfolio_returns = (start..step)
        .map(|col| {
            active_indices
                .iter()
                .map(|idx| weights[*idx] * returns[[*idx, col]])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let portfolio_vol = std_dev(&portfolio_returns) * TRADING_DAYS_PER_YEAR.sqrt();
    if portfolio_vol > 0.0 && portfolio_vol > target_vol {
        let scale = target_vol / portfolio_vol;
        *weights *= scale;
    }
}

fn calculate_max_drawdown(values: &[f64]) -> f64 {
    let mut peak = values[0];
    let mut max_dd = 0.0;
    for &value in values {
        if value > peak {
            peak = value;
        }
        let drawdown = 1.0 - value / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }
    max_dd
}

fn mean_or_zero(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = mean_or_zero(values);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

pub fn calculate_benchmark_stats(closes: &[f64]) -> serde_json::Value {
    if closes.len() < 2 {
        return serde_json::json!({});
    }
    let start_price = closes[0];
    let end_price = closes[closes.len() - 1];
    let total_return = end_price / start_price - 1.0;
    let n_days = closes.len() - 1;
    let ann_return = (end_price / start_price).powf(TRADING_DAYS_PER_YEAR / n_days as f64) - 1.0;

    let returns: Vec<f64> = closes
        .windows(2)
        .map(|window| window[1] / window[0] - 1.0)
        .collect();
    let ann_vol = std_dev(&returns) * TRADING_DAYS_PER_YEAR.sqrt();

    let mut peak = start_price;
    let mut max_dd = 0.0;
    for &price in closes.iter() {
        if price > peak {
            peak = price;
        }
        let drawdown = 1.0 - price / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    serde_json::json!({
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd
    })
}
