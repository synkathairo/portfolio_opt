mod portfolio_opt;

#[tokio::main]
async fn main() {
    // Load .env file from the current directory
    dotenv::dotenv().ok();

    if let Err(e) = portfolio_opt::cli::run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
