use apca::api::v2::order::Order;
use clap::Parser;
use std::collections::HashMap;
use std::fs;

use crate::portfolio_opt::alpaca::{PortfolioClients, YahooCacheOptions};
use crate::portfolio_opt::backtest::{
    DualMomentumOptions, calculate_benchmark_stats, compute_dual_momentum_targets_with_options,
    run_dual_momentum_backtest_with_options,
};
use crate::portfolio_opt::config::OptimizationConfig;
use crate::portfolio_opt::rebalance::{build_order_plan, current_weights};

#[derive(Parser)]
#[command(name = "rust-portfolio")]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "dual-momentum")]
    strategy: String,

    #[arg(long, default_value_t = 252)]
    lookback_days: usize,

    #[arg(long, default_value_t = 0)]
    backtest_days: usize,

    #[arg(long, default_value_t = 21)]
    rebalance_every: usize,

    #[arg(long, default_value_t = 2)]
    top_k: usize,

    #[arg(long, default_value = "equal")]
    dual_momentum_weighting: String,

    #[arg(long, default_value_t = 0.05)]
    softmax_temperature: f64,

    #[arg(long, default_value_t = 0.0)]
    absolute_momentum_threshold: f64,

    #[arg(long)]
    target_vol: Option<f64>,

    #[arg(long, default_value_t = 63)]
    vol_window: usize,

    #[arg(long)]
    max_single_weight: Option<f64>,

    #[arg(long)]
    trailing_stop: Option<f64>,

    #[arg(long, default_value = "yfinance")]
    data_source: String,

    #[arg(long)]
    use_cache: bool,

    #[arg(long)]
    refresh_cache: bool,

    #[arg(long)]
    offline: bool,

    #[arg(long)]
    dry_run: bool,

    #[arg(long)]
    submit: bool,

    #[arg(long, default_value_t = 0.02)]
    rebalance_threshold: f64,

    #[arg(long, default_value_t = 4.0)]
    risk_aversion: f64,

    #[arg(long, default_value_t = 0.35)]
    max_weight: f64,

    #[arg(long, default_value_t = 0.0)]
    min_weight: f64,
}

#[derive(serde::Deserialize)]
struct ModelInputs {
    symbols: Vec<String>,
    asset_classes: HashMap<String, String>,
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.strategy != "dual-momentum" {
        return Err(format!(
            "Rust strategy '{}' is not implemented yet; use dual-momentum",
            args.strategy
        )
        .into());
    }
    if args.trailing_stop.is_some_and(|value| value <= 0.0) {
        return Err("--trailing-stop must be greater than 0".into());
    }
    if args.vol_window == 0 {
        return Err("--vol-window must be greater than 0".into());
    }
    if args.data_source != "yfinance" && args.data_source != "alpaca" {
        return Err(format!(
            "Rust data source '{}' is not implemented yet; use yfinance or alpaca",
            args.data_source
        )
        .into());
    }

    let model_content = fs::read_to_string(&args.model)?;
    let model: ModelInputs = serde_json::from_str(&model_content)?;
    let dm_options = dual_momentum_options(&args);
    let cache_options = yahoo_cache_options(&args);

    if args.backtest_days > 0 {
        let clients = if args.data_source == "alpaca" {
            PortfolioClients::new().await?
        } else {
            PortfolioClients::yahoo_only()
        };

        let total_days = args.lookback_days + args.backtest_days + 1;
        let closes = fetch_closes_for_source(
            &clients,
            &args.data_source,
            &model.symbols,
            total_days,
            cache_options,
        )
        .await?;

        // Filter to symbols with enough data
        let valid_symbols: Vec<String> = model
            .symbols
            .iter()
            .filter(|s| closes.get(*s).map(|v| v.len()).unwrap_or(0) >= total_days)
            .cloned()
            .collect();

        // Align to common trailing history
        let min_len = valid_symbols
            .iter()
            .filter_map(|s| closes.get(s).map(|v| v.len()))
            .min()
            .unwrap_or(0);

        let trim_len = total_days.min(min_len);
        let mut aligned_closes = HashMap::new();
        for s in &valid_symbols {
            if let Some(v) = closes.get(s) {
                if v.len() >= trim_len {
                    aligned_closes.insert(s.clone(), v[v.len() - trim_len..].to_vec());
                }
            }
        }

        if aligned_closes.is_empty() || trim_len < total_days {
            eprintln!("Not enough common history for any symbols.");
            return Ok(());
        }

        let result = run_dual_momentum_backtest_with_options(
            &valid_symbols,
            &aligned_closes,
            &model.asset_classes,
            args.rebalance_every,
            &dm_options,
        )?;

        // Calculate benchmark stats if SPY is available
        let benchmark_stats = if let Some(spy_closes) = aligned_closes.get("SPY") {
            // Slice SPY data to match the aligned backtest window
            // aligned_closes has min_len days. We take the last min_len days of SPY.
            if spy_closes.len() > 1 {
                Some(calculate_benchmark_stats(spy_closes))
            } else {
                None
            }
        } else {
            None
        };

        let mut output = serde_json::json!({
            "symbols": valid_symbols,
            "backtest": {
                "strategy": args.strategy,
                "dual_momentum_weighting": args.dual_momentum_weighting,
                "target_vol": args.target_vol,
                "vol_window": args.vol_window,
                "max_single_weight": args.max_single_weight,
                "trailing_stop": args.trailing_stop,
                "days": args.backtest_days,
                "data_source": args.data_source,
                "rebalance_every": args.rebalance_every,
                "final_value": result.final_value,
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "annualized_volatility": result.annualized_volatility,
                "max_drawdown": result.max_drawdown,
                "rebalance_count": result.rebalance_count,
                "average_turnover": result.average_turnover,
                "daily_values": result.daily_values,
            },
            "latest_target_weights": weights_by_symbol(&valid_symbols, &result.latest_weights),
        });

        if let Some(stats) = benchmark_stats {
            output["benchmarks"] = serde_json::json!({
                "SPY": stats
            });
        }

        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        let clients = PortfolioClients::new().await?;

        // Live / Dry-run mode
        let account = clients.get_account().await?;
        let positions = clients.get_positions().await?;
        let open_orders: Vec<Order> = clients.get_open_orders().await.unwrap_or_default();

        // Fetch history for the signal (lookback + buffer)
        let history_days = args.lookback_days + 50;
        let closes = fetch_closes_for_source(
            &clients,
            &args.data_source,
            &model.symbols,
            history_days,
            cache_options,
        )
        .await?;

        // Check for missing data
        let missing: Vec<_> = model
            .symbols
            .iter()
            .filter(|s| !closes.contains_key(*s))
            .collect();
        if !missing.is_empty() {
            eprintln!("Warning: Missing history for: {:?}", missing);
            return Err("Incomplete history".into());
        }

        // Get latest prices for order sizing
        let latest_prices: std::collections::HashMap<String, f64> = closes
            .iter()
            .filter_map(|(s, v)| v.last().map(|p| (s.clone(), *p)))
            .collect();

        // Compute actual Dual Momentum targets
        let target_weights = compute_dual_momentum_targets_with_options(
            &model.symbols,
            &closes,
            &model.asset_classes,
            &dm_options,
        )?;

        let config = OptimizationConfig {
            risk_aversion: args.risk_aversion,
            min_weight: args.min_weight,
            max_weight: args.max_weight,
            rebalance_threshold: args.rebalance_threshold,
            ..Default::default()
        };
        let plan = build_order_plan(
            &model.symbols,
            &target_weights,
            &account,
            &positions,
            &latest_prices,
            &config,
            Some(
                &open_orders
                    .iter()
                    .map(|o| {
                        serde_json::json!({
                            "symbol": o.symbol,
                            "side": format!("{:?}", o.side)
                        })
                    })
                    .collect::<Vec<_>>(),
            ),
        );

        let result = serde_json::json!({
            "symbols": model.symbols,
            "current_weights": current_weights(&model.symbols, &account, &positions),
            "target_weights": weights_by_symbol(&model.symbols, &target_weights),
            "orders": plan,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);

        if args.submit {
            clients.submit_order_plan(&plan).await?;
        }
    }

    Ok(())
}

fn dual_momentum_options(args: &Args) -> DualMomentumOptions {
    DualMomentumOptions {
        lookback_days: args.lookback_days,
        top_k: args.top_k,
        absolute_threshold: args.absolute_momentum_threshold,
        weighting: args.dual_momentum_weighting.clone(),
        softmax_temperature: args.softmax_temperature,
        target_vol: args.target_vol,
        max_single_weight: args.max_single_weight,
        vol_window: args.vol_window,
        trailing_stop: args.trailing_stop,
    }
}

fn yahoo_cache_options(args: &Args) -> YahooCacheOptions {
    YahooCacheOptions {
        use_cache: args.use_cache,
        refresh_cache: args.refresh_cache,
        offline: args.offline,
    }
}

async fn fetch_closes_for_source(
    clients: &PortfolioClients,
    data_source: &str,
    symbols: &[String],
    lookback_days: usize,
    cache_options: YahooCacheOptions,
) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    match data_source {
        "alpaca" => {
            clients
                .fetch_alpaca_closes_with_options(symbols, lookback_days, cache_options)
                .await
        }
        "yfinance" => {
            clients
                .fetch_yahoo_closes_with_options(symbols, lookback_days, cache_options)
                .await
        }
        _ => Err(format!("Unknown data source: {data_source}").into()),
    }
}

fn weights_by_symbol(symbols: &[String], weights: &[f64]) -> HashMap<String, f64> {
    symbols
        .iter()
        .zip(weights.iter())
        .map(|(symbol, weight)| (symbol.clone(), *weight))
        .collect()
}
