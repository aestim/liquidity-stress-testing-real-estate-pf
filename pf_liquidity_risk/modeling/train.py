import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
from pf_liquidity_risk.config import FIGURES_DIR


# ==========================================
# PF Investment Model Configuration
# ==========================================

@dataclass
class PFConfig:
    """Configuration for Real Estate PF Investment Monte Carlo Simulation"""

    # Capital Structure
    initial_equity: float = 5_600_000_000
    senior_loan: float = 19_000_000_000

    # Operating Cost
    monthly_fixed_cost: float = 150_000_000

    # Monthly Revenue by Phase (min, mode, max)
    # Stochastic distributions for rental income performance
    stabilization_revenue_dist: Tuple[float, float, float] = (30e6, 100e6, 130e6)
    post_court_revenue_dist: Tuple[float, float, float] = (120e6, 150e6, 200e6)
    cap_rate: float = 0.055 # Capitalization rate for income approach

    # Timeline (months)
    completion_target_month: int = 16
    court_opening_month: int = 24
    exit_month: int = 36

    # Interest Rate Scenarios (min, mode, max)
    pre_completion_rate: Tuple[float, float, float] = (0.10, 0.14, 0.18)
    stabilization_rate: Tuple[float, float, float] = (0.08, 0.11, 0.14)
    post_court_rate: Tuple[float, float, float] = (0.05, 0.07, 0.09)

    # Capitalized interest ratio by phase
    capitalized_ratio_construction: float = 1.0
    capitalized_ratio_stabilization: float = 0.4
    capitalized_ratio_exit: float = 0.0

    # Refi uncertainty distributions (min, mode, max)
    target_refi_ltv_dist: Tuple[float, float, float] = (0.70, 0.80, 0.85)


# ==========================================
# PF Survival & Exit Simulation Engine
# ==========================================

class PFInvestmentModel:
    def __init__(self, config: PFConfig):
        self.cfg = config

    def _get_rate_params(self, phase: str) -> Tuple[float, float, float]:
        return {
            "construction": self.cfg.pre_completion_rate,
            "stabilization": self.cfg.stabilization_rate,
            "exit": self.cfg.post_court_rate
        }[phase]

    def _get_capitalized_ratio(self, phase: str) -> float:
        return {
            "construction": self.cfg.capitalized_ratio_construction,
            "stabilization": self.cfg.capitalized_ratio_stabilization,
            "exit": self.cfg.capitalized_ratio_exit
        }[phase]

    def simulate_path(self) -> Dict:
        equity = self.cfg.initial_equity
        principal = self.cfg.senior_loan

        # Sampling revenue performance for this specific simulation path
        sampled_stab_rev = np.random.triangular(*self.cfg.stabilization_revenue_dist)
        sampled_post_rev = np.random.triangular(*self.cfg.post_court_revenue_dist)

        # Uncertainty in construction completion timing
        completion_month = self.cfg.completion_target_month + int(
            np.random.triangular(0, 2, 6)
        )

        for m in range(1, self.cfg.exit_month + 1):
            if m < completion_month:
                phase = "construction"
                revenue = 0
            elif m < self.cfg.court_opening_month:
                phase = "stabilization"
                revenue = sampled_stab_rev
            else:
                phase = "exit"
                revenue = sampled_post_rev

            # Interest calculation
            annual_rate = np.random.triangular(*self._get_rate_params(phase))
            monthly_rate = annual_rate / 12
            interest = principal * monthly_rate

            cap_ratio = self._get_capitalized_ratio(phase)
            paid_interest = interest * (1 - cap_ratio)
            capitalized_interest = interest * cap_ratio
            
            # Loan balance increases due to capitalized interest
            principal += capitalized_interest

            # Net cash flow calculation
            net_cash_flow = revenue - (self.cfg.monthly_fixed_cost + paid_interest)
            
            # Cash Sweep Logic: Use surplus cash to pay down principal
            if net_cash_flow > 0:
                principal -= net_cash_flow
                if principal < 0:
                    equity += abs(principal)
                    principal = 0
            else:
                # Use equity buffer to cover deficits
                equity += net_cash_flow

            # Check for insolvency
            if equity <= 0:
                return {"status": "default", "month": m}

            # Refinancing decision (Month 24)
            if m == self.cfg.court_opening_month:
                ltv_limit = np.random.triangular(*self.cfg.target_refi_ltv_dist)
                
                # Value derived from sampled revenue using Income Approach
                implied_asset_value = (revenue * 12) / self.cfg.cap_rate
                max_refi = implied_asset_value * ltv_limit

                if principal > max_refi:
                    return {"status": "refi_fail", "month": m}

            # Final Exit (Month 36)
            if m == self.cfg.exit_month:
                # Final valuation based on stabilized rental income
                final_asset_value = (revenue * 12) / self.cfg.cap_rate
                exit_equity = final_asset_value - principal
                
                if exit_equity > 0:
                    irr = (exit_equity / self.cfg.initial_equity) ** (12 / m) - 1
                else:
                    irr = -1.0 # Wipeout scenario

                return {
                    "status": "exit",
                    "month": m,
                    "final_equity": max(0, exit_equity),
                    "irr": irr
                }

        return {"status": "survived_no_exit"}


# ==========================================
# Monte Carlo Execution
# ==========================================

def run_simulation(iterations: int = 30000):
    cfg = PFConfig()
    model = PFInvestmentModel(cfg)
    results = [model.simulate_path() for _ in range(iterations)]
    df = pd.DataFrame(results)
    return df, cfg


# ==========================================
# Result Visualization
# ==========================================

def plot_results(df: pd.DataFrame, cfg: PFConfig, filename: str = "pf_exit_simulation.png"):
    status_counts = df["status"].value_counts()

    if "irr" in df.columns:
        irr_values = df.loc[df["status"] == "exit", "irr"]
    else:
        irr_values = pd.Series(dtype=float)

    plt.style.use("seaborn-v0_8-muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Outcomes plot
    status_counts.plot(kind="bar", ax=axes[0], color=['#E74C3C', '#F1C40F', '#2ECC71', '#3498DB'])
    axes[0].set_title("Project Outcome Distribution", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Frequency")
    plt.setp(axes[0].get_xticklabels(), rotation=0)

    # IRR Histogram
    if len(irr_values) > 0:
        irr_values.plot(kind="hist", bins=40, ax=axes[1], title="Equity IRR Distribution")
        axes[1].set_xlabel("IRR")
    else:
        axes[1].text(0.5, 0.5, "No Exit → No IRR Data",
                     ha="center", va="center", fontsize=12)
        axes[1].set_axis_off()

    plt.tight_layout()
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Visualization saved to: {save_path}")


# ==========================================
# Main Entry
# ==========================================

def main():
    df, cfg = run_simulation(30000)

    print("\n========== PF INVESTMENT SIMULATION RESULTS ==========\n")
    print(df["status"].value_counts(normalize=True))

    print("\nIRR Summary:")
    if "irr" in df.columns:
        irr_series = df.loc[df["status"] == "exit", "irr"]
        if len(irr_series) > 0:
            print(irr_series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
        else:
            print("No exit cases → IRR not available")

    plot_results(df, cfg)


if __name__ == "__main__":
    main()