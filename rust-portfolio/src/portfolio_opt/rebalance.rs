use std::collections::HashMap;
use crate::portfolio_opt::types::{AccountSnapshot, OrderPlan, Position};
use crate::portfolio_opt::config::OptimizationConfig;

pub fn current_weights(
    symbols: &[String],
    account: &AccountSnapshot,
    positions: &[Position],
) -> HashMap<String, f64> {
    let by_symbol: HashMap<_, _> = positions.iter()
        .map(|p| (p.symbol.clone(), p))
        .collect();

    symbols.iter().map(|s| {
        let val = by_symbol.get(s)
            .map(|p| p.market_value)
            .unwrap_or(0.0);
        (s.clone(), if account.equity > 0.0 { val / account.equity } else { 0.0 })
    }).collect()
}

pub fn build_order_plan(
    symbols: &[String],
    target_weights: &[f64],
    account: &AccountSnapshot,
    positions: &[Position],
    latest_prices: &HashMap<String, f64>,
    config: &OptimizationConfig,
    open_orders: Option<&[serde_json::Value]>,
) -> Vec<OrderPlan> {
    let mut weights_now = current_weights(symbols, account, positions);

    // Adjust weights for pending orders
    if let Some(orders) = open_orders {
        for order in orders {
            if let (Some(symbol), Some(side)) = (order.get("symbol").and_then(|v| v.as_str()), order.get("side")) {
                if let Some(_pos) = symbols.iter().position(|s| s == symbol) {
                    let qty = order.get("qty").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let adjusted_qty = if side == "sell" { -qty } else { qty };
                    let price = latest_prices.get(symbol).copied().unwrap_or(0.0);
                    let notional = adjusted_qty * price;
                    if account.equity > 0.0 {
                        *weights_now.get_mut(symbol).unwrap() += notional / account.equity;
                    }
                }
            }
        }
    }

    let mut plans = Vec::new();
    for (symbol, &target_weight) in symbols.iter().zip(target_weights.iter()) {
        let current_weight = weights_now.get(symbol).copied().unwrap_or(0.0);
        let delta = target_weight - current_weight;
        let notional = delta.abs() * account.equity;

        if delta.abs() < config.rebalance_threshold {
            continue;
        }
        if latest_prices.get(symbol).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }

        plans.push(OrderPlan {
            symbol: symbol.clone(),
            current_weight,
            target_weight,
            delta_weight: delta,
            side: if delta > 0.0 { "buy".into() } else { "sell".into() },
            notional_usd: notional.round(),
        });
    }
    plans
}
