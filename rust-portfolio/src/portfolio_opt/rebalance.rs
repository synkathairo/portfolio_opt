use crate::portfolio_opt::config::OptimizationConfig;
use crate::portfolio_opt::types::{AccountSnapshot, OrderPlan, PendingOrderExposure, Position};
use std::collections::HashMap;

pub fn current_weights(
    symbols: &[String],
    account: &AccountSnapshot,
    positions: &[Position],
) -> HashMap<String, f64> {
    let by_symbol: HashMap<_, _> = positions.iter().map(|p| (p.symbol.clone(), p)).collect();

    symbols
        .iter()
        .map(|s| {
            let val = by_symbol.get(s).map(|p| p.market_value).unwrap_or(0.0);
            (
                s.clone(),
                if account.equity > 0.0 {
                    val / account.equity
                } else {
                    0.0
                },
            )
        })
        .collect()
}

pub fn build_order_plan(
    symbols: &[String],
    target_weights: &[f64],
    account: &AccountSnapshot,
    positions: &[Position],
    latest_prices: &HashMap<String, f64>,
    config: &OptimizationConfig,
    open_orders: Option<&[PendingOrderExposure]>,
) -> Vec<OrderPlan> {
    let mut weights_now = current_weights(symbols, account, positions);

    // Adjust weights for pending orders
    if let Some(orders) = open_orders {
        for order in orders {
            if symbols.iter().any(|symbol| symbol == &order.symbol) {
                let price = latest_prices.get(&order.symbol).copied().unwrap_or(0.0);
                let notional = order.signed_remaining_qty * price;
                if account.equity > 0.0 {
                    *weights_now.get_mut(&order.symbol).unwrap() += notional / account.equity;
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
            side: if delta > 0.0 {
                "buy".into()
            } else {
                "sell".into()
            },
            notional_usd: notional.round(),
        });
    }
    plans
}

#[cfg(test)]
mod tests {
    use super::build_order_plan;
    use crate::portfolio_opt::config::OptimizationConfig;
    use crate::portfolio_opt::types::{AccountSnapshot, PendingOrderExposure, Position};
    use std::collections::HashMap;

    #[test]
    fn pending_buy_and_sell_exposure_is_included_in_current_weight() {
        let symbols = vec!["AAA".to_string(), "BBB".to_string()];
        let prices = HashMap::from([("AAA".to_string(), 100.0), ("BBB".to_string(), 50.0)]);
        let pending = vec![
            PendingOrderExposure {
                symbol: "AAA".to_string(),
                signed_remaining_qty: 5.0,
            },
            PendingOrderExposure {
                symbol: "BBB".to_string(),
                signed_remaining_qty: -10.0,
            },
        ];
        let config = OptimizationConfig {
            rebalance_threshold: 0.001,
            ..Default::default()
        };

        let plan = build_order_plan(
            &symbols,
            &[0.5, 0.0],
            &AccountSnapshot { equity: 10_000.0 },
            &[
                Position {
                    symbol: "AAA".to_string(),
                    qty: 5.0,
                    market_value: 500.0,
                },
                Position {
                    symbol: "BBB".to_string(),
                    qty: 10.0,
                    market_value: 500.0,
                },
            ],
            &prices,
            &config,
            Some(&pending),
        );

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].symbol, "AAA");
        assert_eq!(plan[0].side, "buy");
        assert_eq!(plan[0].notional_usd, 4_000.0);
    }
}
