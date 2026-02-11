from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from pf_liquidity_risk.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR 

app = typer.Typer()

def run_liquidity_simulation(iterations: int, months: int):
    """
    Refined Monte Carlo simulation for Project Financing (PF) equity exhaustion risk.
    Simulates compounding interest, realistic rate skew, and revenue timing gaps.
    """
    initial_equity = 4_900_000_000  
    base_loan_principal = 19_000_000_000 
    base_monthly_burn = 150_000_000 
    
    months_until_insolvency = []

    for _ in tqdm(range(iterations), desc="Simulating (Distressed Model)"):
        current_equity = initial_equity
        current_principal = base_loan_principal 
        is_insolvent = False
        
        # 1. Stochastic Dead Zone (Construction Delay + Operational Gap)
        # Lognormal distribution for delay, Normal distribution for gap
        construction_delay = np.random.lognormal(mean=np.log(2), sigma=0.5)
        operational_gap = np.random.normal(loc=14, scale=2)
        dead_zone = int(construction_delay + operational_gap)
        
        # 2. Realistic Interest Rates (Triangular Distribution)
        # Skewed towards 14% (Mode) based on current market conditions
        annual_rate = np.random.triangular(left=0.09, mode=0.14, right=0.16)
        monthly_rate = annual_rate / 12
        
        # Ratio of unpaid interest capitalized into principal (Conservative: 50%)
        capitalization_ratio = 0.5 

        for m in range(1, months + 1):
            # Exit loop if the project survives beyond the 'Dead Zone'
            if m > dead_zone:
                break 

            # Calculate interest expense based on the updated principal
            interest_expense = current_principal * monthly_rate
            
            # Total monthly cash outflow
            total_outflow = interest_expense + base_monthly_burn
            current_equity -= total_outflow
            
            # 3. Compounding Risk: Add a portion of interest to the principal
            current_principal += (interest_expense * capitalization_ratio)

            # Check for insolvency
            if current_equity <= 0:
                months_until_insolvency.append(m)
                is_insolvent = True
                break
        
        if not is_insolvent:
            # Case where the project survives the simulation horizon
            months_until_insolvency.append(months + 1)

    return np.array(months_until_insolvency)

@app.command()
def main(
    iterations: int = typer.Option(10000, help="Number of Monte Carlo trials"),
    months: int = typer.Option(30, help="Total simulation horizon in months"),
    output_filename: str = typer.Option("equity_runway_dist.png", help="Name of the output plot file"),
):
    output_path = FIGURES_DIR / output_filename
    logger.info(f"Execution started: {iterations} trials / {months} months.")

    try:
        results = run_liquidity_simulation(iterations, months)
        
        plt.style.use('seaborn-v0_8-muted')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # --- FIX FOR THE BINS TYPE ERROR ---
        # Convert numpy ndarray to a standard Python list to satisfy type checkers
        bin_edges = np.arange(0.5, months + 1.5, 1).tolist()
        
        counts, bins, _ = ax.hist(
            results[results <= months], 
            bins=bin_edges, # Now passed as a list[float]
            color='#C0392B', 
            alpha=0.75, 
            label='Insolvency Cases', 
            edgecolor='white', 
            linewidth=0.5
        )
        # -----------------------------------
        
        survival_count = np.sum(results > months)
        max_freq = max(np.max(counts) if len(counts) > 0 else 0, survival_count)
        
        # Maintain 40% headroom at the top to prevent overlap with stats box
        ax.set_ylim(0, max_freq * 1.4) 

        # 2. Survival Bar
        if survival_count > 0:
            ax.bar(months + 2, survival_count, color='#27AE60', width=0.8,
                   label='Survival (Post-Dead Zone)', edgecolor='black')
        
        # 3. Milestone & Labels
        ax.axvline(x=14, color='#2E86C1', linestyle='--', linewidth=3, 
                   label='Target Milestone (M14)', zorder=5)
        
        ax.set_title(f'Liquidity Exhaustion Risk Analysis (n={iterations})', 
                     fontsize=16, fontweight='bold', pad=30)
        ax.set_xlabel('Month of Exhaustion', fontsize=12, labelpad=10)
        ax.set_ylabel('Frequency (Trials)', fontsize=12)

        # 4. Statistics Box (Top-Left)
        insolvency_rate = (len(results[results <= months]) / iterations) * 100
        avg_insolvency = np.mean(results[results <= months]) if len(results[results <= months]) > 0 else 0
        
        stats_text = (
            f"▼ KEY RISK METRICS\n"
            f" Insolvency Rate: {insolvency_rate:.1f}%\n"
            f" Avg. Exhaustion: Month {avg_insolvency:.1f}\n\n"
            f"▼ ASSUMPTIONS\n"
            f" Interest Rate : Tri(9%, 14%, 16%)\n"
            f" Capitalization: 50%"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#D5DBDB')
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')

        # 5. Legend & Grid
        ax.legend(loc='upper right', frameon=True, shadow=True)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        # 6. X-Axis Ticks (Including the "Safe" zone conversion)
        tick_positions = list(range(0, months + 1, 2)) + [months + 2]
        tick_labels = [str(x) if isinstance(x, int) else x for x in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.success(f"Figure saved successfully: {output_path}")

    except Exception as e:
        logger.error(f"Error generating figure: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()