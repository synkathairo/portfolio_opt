use apca::api::v2::account;
use apca::api::v2::order::{self, CreateReqInit, Amount, Type, TimeInForce, Side, Order};
use apca::api::v2::positions;
use apca::api::v2::orders::{self, ListReq, Status};
use apca::{ApiInfo, Client as AlpacaClient};
use num_decimal::Num;
use std::collections::HashMap;
use yfinance_rs::{Interval, Range, Ticker, YfClient};
use yfinance_rs::core::conversions::money_to_f64;

use crate::portfolio_opt::types::{AccountSnapshot, OrderPlan, Position};

fn num_to_f64(num: Num) -> f64 {
    num.to_string().parse().unwrap_or(0.0)
}

pub struct PortfolioClients {
    alpaca: AlpacaClient,
    yf: YfClient,
}

impl PortfolioClients {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let api_info = ApiInfo::from_env()?;
        Ok(Self {
            alpaca: AlpacaClient::new(api_info),
            yf: YfClient::default(),
        })
    }

    pub async fn get_account(&self) -> Result<AccountSnapshot, Box<dyn std::error::Error>> {
        let info = self.alpaca.issue::<account::Get>(&()).await?;
        Ok(AccountSnapshot {
            equity: num_to_f64(info.equity),
        })
    }

    pub async fn get_positions(&self) -> Result<Vec<Position>, Box<dyn std::error::Error>> {
        let positions = self.alpaca.issue::<positions::List>(&()).await?;
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
        let req = ListReq {
            status: Status::Open,
            ..Default::default()
        };
        self.alpaca.issue::<orders::List>(&req).await.map_err(|e| e.into())
    }

    pub async fn submit_order_plan(&self, plans: &[OrderPlan]) -> Result<(), Box<dyn std::error::Error>> {
        for plan in plans {
            let side = if plan.side == "buy" { Side::Buy } else { Side::Sell };
            
            let req_init = CreateReqInit {
                type_: Type::Market,
                time_in_force: TimeInForce::Day,
                ..Default::default()
            };

            let notional_val: Num = format!("{}", plan.notional_usd).parse().unwrap();
            let req = req_init.init(
                plan.symbol.as_str(), 
                side, 
                Amount::Notional { notional: notional_val }
            );

            if let Err(e) = self.alpaca.issue::<order::Create>(&req).await {
                eprintln!("Failed to submit order for {}: {}", plan.symbol, e);
            }
        }
        Ok(())
    }

    pub async fn fetch_yahoo_closes(&self, symbols: &[String], period_days: usize) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
        let mut closes = HashMap::new();
        
        // Map days to Yahoo Range
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
                let prices: Vec<f64> = history.iter()
                    .map(|c| money_to_f64(&c.close))
                    .collect();
                if !prices.is_empty() {
                    closes.insert(symbol.clone(), prices);
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Ok(closes)
    }
}
