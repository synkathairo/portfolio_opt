use std::collections::HashMap;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AlpacaConfig {
    pub api_key: String,
    pub api_secret: String,
    pub base_url: String,
    pub data_url: String,
}

impl AlpacaConfig {
    #[allow(dead_code)]
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        dotenv::dotenv().ok();
        Ok(Self {
            api_key: std::env::var("APCA_API_KEY_ID")?,
            api_secret: std::env::var("APCA_API_SECRET_KEY")?,
            base_url: std::env::var("APCA_API_BASE_URL")
                .unwrap_or_else(|_| "https://paper-api.alpaca.markets".into()),
            data_url: std::env::var("APCA_API_DATA_URL")
                .unwrap_or_else(|_| "https://data.alpaca.markets".into()),
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub risk_aversion: f64,
    pub min_weight: f64,
    pub max_weight: f64,
    pub rebalance_threshold: f64,
    pub turnover_penalty: f64,
    pub force_full_investment: bool,
    pub min_cash_weight: f64,
    pub max_turnover: Option<f64>,
    pub min_invested_weight: f64,
    pub class_min_weights: HashMap<String, f64>,
    pub class_max_weights: HashMap<String, f64>,
}

impl OptimizationConfig {
    pub fn new() -> Self {
        Self {
            risk_aversion: 4.0,
            min_weight: 0.0,
            max_weight: 0.35,
            rebalance_threshold: 0.02,
            turnover_penalty: 0.02,
            force_full_investment: false,
            min_cash_weight: 0.0,
            max_turnover: None,
            min_invested_weight: 0.0,
            class_min_weights: HashMap::new(),
            class_max_weights: HashMap::new(),
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::new()
    }
}
