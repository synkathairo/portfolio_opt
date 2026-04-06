use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub qty: f64,
    pub market_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub equity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderPlan {
    pub symbol: String,
    pub current_weight: f64,
    pub target_weight: f64,
    pub delta_weight: f64,
    pub side: String,
    pub notional_usd: f64,
}
