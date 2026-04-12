import json
import subprocess
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

today = str(date.today())

MODEL = "examples/nasdaq100_sp500_sector_universe_b2016filtered.json"
TITLE_NAME_VAR = "nasdaq100_sp500_sector_universe_b2016filtered-voltarget"
LOOKBACK = 252
BACKTEST_DAYS = 252 * 9
REBALANCE_EVERY = 1
TOP_K = 2
DUAL_MOMENTUM_WEIGHTING = "equal"
DATA_SOURCE = "yfinance"
TIMEOUT_LEN = 2400
YFINANCE_MAX_WORKERS = 1
YFINANCE_RETRY_DELAY = 5.0
REFRESH_CACHE = False

TARGET_VOLS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
VOL_WINDOWS = [21, 42, 63, 84, 126, 252]

FIGURE_NAME = (
    f"vol_window_sweep_{TITLE_NAME_VAR}_{BACKTEST_DAYS}_{LOOKBACK}_{today}"
)
CSV_NAME = f"{FIGURE_NAME}.csv"


def build_command(
    target_vol: float,
    vol_window: int,
    *,
    offline: bool,
    refresh_cache: bool,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "portfolio-opt",
        "--model",
        MODEL,
        "--strategy",
        "dual-momentum",
        "--lookback-days",
        str(LOOKBACK),
        "--backtest-days",
        str(BACKTEST_DAYS),
        "--rebalance-every",
        str(REBALANCE_EVERY),
        "--top-k",
        str(TOP_K),
        "--dual-momentum-weighting",
        DUAL_MOMENTUM_WEIGHTING,
        "--target-vol",
        str(target_vol),
        "--vol-window",
        str(vol_window),
        "--data-source",
        DATA_SOURCE,
        "--use-cache",
        "--yfinance-max-workers",
        str(YFINANCE_MAX_WORKERS),
        "--yfinance-retry-delay",
        str(YFINANCE_RETRY_DELAY),
    ]
    if offline:
        cmd.append("--offline")
    if refresh_cache:
        cmd.append("--refresh-cache")
    return cmd


def run_command(cmd: list[str], label: str) -> dict | None:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_LEN, check=False
        )
        if result.returncode != 0:
            print(f"Error for {label}: {result.stderr.strip()}")
            return None
        data = json.loads(result.stdout)
        return data["backtest"]
    except Exception as exc:
        print(f"Error for {label}: {exc}")
        return None


def warm_yfinance_cache() -> bool:
    print("Warming yfinance cache...")
    cmd = build_command(
        TARGET_VOLS[0],
        VOL_WINDOWS[0],
        offline=False,
        refresh_cache=REFRESH_CACHE,
    )
    return run_command(cmd, "cache warmup") is not None


def run_backtest(target_vol: float, vol_window: int) -> dict | None:
    cmd = build_command(target_vol, vol_window, offline=True, refresh_cache=False)
    return run_command(cmd, f"target_vol={target_vol}, vol_window={vol_window}")


def main() -> None:
    print("Running volatility-target grid search...")
    if not warm_yfinance_cache():
        raise SystemExit(
            "Could not warm yfinance cache. Yahoo may be rate limiting this IP; "
            "wait and retry, or reduce the universe size."
        )
    rows = []
    for target_vol in TARGET_VOLS:
        for vol_window in VOL_WINDOWS:
            print(f"Running target_vol={target_vol}, vol_window={vol_window}...")
            res = run_backtest(target_vol, vol_window)
            if not res:
                continue

            ann_ret = float(res.get("annualized_return", 0.0))
            ann_vol = float(res.get("annualized_volatility", 0.0))
            max_dd = abs(float(res.get("max_drawdown", 0.0)))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            calmar = ann_ret / max_dd if max_dd > 0 else 0.0

            rows.append(
                {
                    "Target Vol": target_vol,
                    "Vol Window": vol_window,
                    "Annualized Return": ann_ret,
                    "Annualized Volatility": ann_vol,
                    "Sharpe Ratio": sharpe,
                    "Calmar Ratio": calmar,
                    "Max Drawdown": float(res.get("max_drawdown", 0.0)),
                    "Average Turnover": float(res.get("average_turnover", 0.0)),
                    "Rebalance Count": int(res.get("rebalance_count", 0)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No successful backtests; no figure generated.")

    df.to_csv(CSV_NAME, index=False)
    print(f"Saved {CSV_NAME}")

    df_return = df.pivot(
        index="Vol Window", columns="Target Vol", values="Annualized Return"
    )
    df_vol = df.pivot(
        index="Vol Window", columns="Target Vol", values="Annualized Volatility"
    )
    df_dd = df.pivot(index="Vol Window", columns="Target Vol", values="Max Drawdown")
    df_turn = df.pivot(
        index="Vol Window", columns="Target Vol", values="Average Turnover"
    )
    df_sharpe = df.pivot(index="Vol Window", columns="Target Vol", values="Sharpe Ratio")
    df_calmar = df.pivot(index="Vol Window", columns="Target Vol", values="Calmar Ratio")

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(
        (
            f"{TITLE_NAME_VAR} Vol Target Sweep "
            f"(Lookback={LOOKBACK}, Days={BACKTEST_DAYS}, TopK={TOP_K}, "
            f"Rebalance={REBALANCE_EVERY})"
        ),
        fontsize=16,
    )

    sns.heatmap(
        df_return,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=axes[0, 0],
        cbar_kws={"label": "Return"},
    )
    axes[0, 0].set_title("Annualized Return")

    sns.heatmap(
        df_vol,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        ax=axes[0, 1],
        cbar_kws={"label": "Vol"},
    )
    axes[0, 1].set_title("Annualized Volatility")

    sns.heatmap(
        df_dd,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        ax=axes[1, 0],
        cbar_kws={"label": "Drawdown"},
    )
    axes[1, 0].set_title("Max Drawdown")

    sns.heatmap(
        df_turn,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        ax=axes[1, 1],
        cbar_kws={"label": "Turnover"},
    )
    axes[1, 1].set_title("Average Turnover")

    sns.heatmap(
        df_sharpe,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[2, 0],
        cbar_kws={"label": "Sharpe"},
    )
    axes[2, 0].set_title("Sharpe Ratio")

    sns.heatmap(
        df_calmar,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[2, 1],
        cbar_kws={"label": "Calmar"},
    )
    axes[2, 1].set_title("Calmar Ratio")

    plt.tight_layout()
    plt.savefig(f"{FIGURE_NAME}.png", dpi=150)
    print(f"Saved {FIGURE_NAME}.png")


if __name__ == "__main__":
    main()
