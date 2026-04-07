use apca::api::v2::order::Order;
use clap::Parser;
use std::collections::HashMap;
use std::fs;

use crate::portfolio_opt::alpaca::PortfolioClients;
use crate::portfolio_opt::backtest::{
    calculate_benchmark_stats, compute_dual_momentum_targets, run_dual_momentum_backtest,
};
use crate::portfolio_opt::config::OptimizationConfig;
use crate::portfolio_opt::rebalance::{build_order_plan, current_weights};

#[derive(Parser)]
#[command(name = "rust-portfolio")]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "mean-variance")]
    strategy: String,

    #[arg(long, default_value_t = 252)]
    lookback_days: usize,

    #[arg(long, default_value_t = 0)]
    backtest_days: usize,

    #[arg(long, default_value_t = 21)]
    rebalance_every: usize,

    #[arg(long, default_value_t = 2)]
    top_k: usize,

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
    let model_content = fs::read_to_string(&args.model)?;
    let model: ModelInputs = serde_json::from_str(&model_content)?;

    let clients = PortfolioClients::new().await?;

    if args.backtest_days > 0 {
        // Backtest mode - use Yahoo Finance for historical data
        let total_days = args.lookback_days + args.backtest_days;
        let closes = clients
            .fetch_yahoo_closes(&model.symbols, total_days)
            .await?;

        // Filter to symbols with enough data
        let valid_symbols: Vec<String> = model
            .symbols
            .iter()
            .filter(|s| closes.get(*s).map(|v| v.len()).unwrap_or(0) > args.lookback_days)
            .cloned()
            .collect();

        // Align to common trailing history
        let min_len = valid_symbols
            .iter()
            .filter_map(|s| closes.get(s).map(|v| v.len()))
            .min()
            .unwrap_or(0);

        let mut aligned_closes = HashMap::new();
        for s in &valid_symbols {
            if let Some(v) = closes.get(s) {
                if v.len() >= min_len {
                    aligned_closes.insert(s.clone(), v[v.len() - min_len..].to_vec());
                }
            }
        }

        if aligned_closes.is_empty() || min_len < args.lookback_days + 1 {
            eprintln!("Not enough common history for any symbols.");
            return Ok(());
        }

        let result = run_dual_momentum_backtest(
            &valid_symbols,
            &aligned_closes,
            &model.asset_classes,
            args.lookback_days,
            args.rebalance_every,
            args.top_k,
            0.0,
        )?;

        // Calculate benchmark stats if SPY is available
        let benchmark_stats = if let Some(spy_closes) = closes.get("SPY") {
            // Slice SPY data to match the aligned backtest window
            // aligned_closes has min_len days. We take the last min_len days of SPY.
            let min_len = valid_symbols
                .iter()
                .filter_map(|s| closes.get(s).map(|v| v.len()))
                .min()
                .unwrap_or(0);

            if spy_closes.len() >= min_len && min_len > 1 {
                let spy_slice = &spy_closes[spy_closes.len() - min_len..];
                Some(calculate_benchmark_stats(spy_slice))
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
                "days": args.backtest_days,
                "rebalance_every": args.rebalance_every,
                "final_value": result.final_value,
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "annualized_volatility": result.annualized_volatility,
                "max_drawdown": result.max_drawdown,
                "rebalance_count": result.rebalance_count,
                "average_turnover": result.average_turnover,
            },
        });

        if let Some(stats) = benchmark_stats {
            output["benchmarks"] = serde_json::json!({
                "SPY": stats
            });
        }

        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // Live / Dry-run mode
        let account = clients.get_account().await?;
        let positions = clients.get_positions().await?;
        let open_orders: Vec<Order> = clients.get_open_orders().await.unwrap_or_default();

        // Fetch history for the signal (lookback + buffer)
        let history_days = args.lookback_days + 50;
        let closes = clients
            .fetch_yahoo_closes(&model.symbols, history_days)
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
        let target_weights = compute_dual_momentum_targets(
            &model.symbols,
            &closes,
            &model.asset_classes,
            args.lookback_days,
            args.top_k,
            0.0,
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
            "target_weights": target_weights,
            "orders": plan,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);

        if args.submit {
            clients.submit_order_plan(&plan).await?;
        }
    }

    Ok(())
}
