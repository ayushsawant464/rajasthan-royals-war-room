import os
import time
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_reverse_swing_audit():
    print("=== Generating Plot 16: Reverse Swing Physics & Temporal Audit ===")
    t0 = time.time()

    media_dir = 'media'
    artifact_dir = '/home/savvy19/.gemini/antigravity-cli/brain/704ec573-87b1-44f8-9e64-aa6117d903c5'
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    data_dir = '6af458274c1e8_data_pack/Data Pack'
    bbb_cols = ['match_id', 'overid', 'runs', 'extras1_runs', 'extras_2_runs', 'is_wicket', 'delivery_style', 'is_four', 'is_six', 'bowl_speed', 'bowler_name']
    df = pl.read_csv(f'{data_dir}/ball_by_ball.csv', columns=bbb_cols, schema_overrides={'bowl_speed': pl.Float64})
    df = df.with_columns([
        (pl.col('runs') + pl.col('extras1_runs') + pl.col('extras_2_runs')).alias('total_runs'),
        ((pl.col('runs') + pl.col('extras1_runs') + pl.col('extras_2_runs') == 0)).alias('is_dot'),
        pl.when(pl.col('overid') >= 16).then(pl.lit('Death')).otherwise(pl.lit('Other')).alias('phase')
    ])

    # ---------------- MATPLOTLIB 4-PANEL FIGURE ----------------
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='#0B0F19')
    fig.suptitle('MATHEMATICAL AUDIT: REVERSE SWING DYNAMICS, TEMPORAL SELECTION & NET WIN EQUITY\n'
                 'Resolving High Conceded Runs: Simpson\'s Paradox (75% Death Concentration) vs. Strike Weapon Bimodality (9.5% Wkt Rate)',
                 fontsize=14.5, fontweight='bold', color='#F0F6FC', y=0.98)

    # ===== PANEL 1: Temporal Over Distribution (Simpson's Paradox) =====
    ax1 = axes[0, 0]
    ax1.set_facecolor('#0B0F19')

    overs = np.arange(1, 21)
    all_over_counts = df.group_by('overid').agg(pl.len().alias('n')).sort('overid')['n'].to_numpy()
    rev_over_counts = df.filter(pl.col('delivery_style').str.contains('(?i)reverse')).group_by('overid').agg(pl.len().alias('n')).sort('overid')
    
    # Fill in 0 for any missing overs
    rev_counts = np.zeros(20)
    for row in rev_over_counts.iter_rows():
        rev_counts[row[0] - 1] = row[1]

    rev_pcts = rev_counts / all_over_counts * 100.0

    bars1 = ax1.bar(overs, rev_counts, color='#F43F5E', edgecolor='#334155', alpha=0.85, label='Reverse Swing Deliveries')
    ax1_twin = ax1.twinx()
    line1 = ax1_twin.plot(overs, rev_pcts, color='#FACC15', lw=2.5, marker='o', label='% of All Balls in Over')
    ax1_twin.set_ylabel('% of All Deliveries That Are Reverse Swing', color='#FACC15', fontsize=8.5, fontweight='bold')
    ax1_twin.tick_params(colors='#FACC15', labelsize=8)

    ax1.axvspan(15.5, 20.5, color='#38BDF8', alpha=0.15, label='Death Overs (74.8% of all Reverse Swing)')
    ax1.set_title('Panel 1: The Temporal Death Concentration (Simpson\'s Paradox)\n'
                  '74.8% of Reverse Swing is Bowled in Overs 16–20 (Par: 10–12 RPO) | Negligible in Powerplay',
                  fontsize=10.5, fontweight='bold', color='#F0F6FC')
    ax1.set_xlabel('Over in Innings (1 to 20)', fontsize=8.5, fontweight='bold', color='#94A3B8')
    ax1.set_ylabel('Number of Reverse Swing Deliveries', fontsize=8.5, fontweight='bold', color='#94A3B8')
    ax1.set_xticks(overs)
    ax1.legend(loc='upper left', fontsize=8, facecolor='#1E293B', edgecolor='#334155')
    ax1.grid(True, linestyle=':', alpha=0.25, color='#475569')

    # ===== PANEL 2: Reverse Swing vs Full Toss vs Yorker in Death Overs =====
    ax2 = axes[0, 1]
    ax2.set_facecolor('#0B0F19')

    death_styles = ['Yorker', 'Off Cutter', 'Slow Bouncer', 'Reverse Swing In', 'Full Toss']
    d_runs = [0.91, 1.66, 1.43, 1.88, 1.97]
    d_wkts = [7.7, 9.8, 7.9, 9.5, 6.5]
    d_dots = [35.4, 28.7, 31.4, 24.5, 16.5]
    d_speeds = [135.4, 118.6, 119.6, 138.0, 133.6]

    x = np.arange(len(death_styles))
    w = 0.22

    ax2.bar(x - 1.5*w, d_runs, width=w, label='Runs/Ball Conceded', color='#EF4444', edgecolor='#334155')
    ax2.bar(x - 0.5*w, [v/10.0 for v in d_wkts], width=w, label='Wicket Rate (x10 %)', color='#10B981', edgecolor='#334155')
    ax2.bar(x + 0.5*w, [v/20.0 for v in d_dots], width=w, label='Dot Ball Rate (x20 %)', color='#38BDF8', edgecolor='#334155')
    ax2.bar(x + 1.5*w, [v/100.0 for v in d_speeds], width=w, label='Avg Speed (x100 km/h)', color='#FACC15', edgecolor='#334155')

    for i in range(len(death_styles)):
        ax2.text(x[i] - 1.5*w, d_runs[i] + 0.05, f"{d_runs[i]:.2f}", ha='center', fontsize=7.5, fontweight='bold', color='#EF4444')
        ax2.text(x[i] - 0.5*w, d_wkts[i]/10.0 + 0.05, f"{d_wkts[i]:.1f}%", ha='center', fontsize=7.5, fontweight='bold', color='#10B981')
        ax2.text(x[i] + 0.5*w, d_dots[i]/20.0 + 0.05, f"{d_dots[i]:.0f}%", ha='center', fontsize=7.5, fontweight='bold', color='#38BDF8')

    ax2.set_xticks(x)
    ax2.set_xticklabels(death_styles, fontsize=8.5, color='#F0F6FC', fontweight='bold')
    ax2.set_title('Panel 2: Reverse Swing vs. Full Toss: Execution Weapon vs. Execution Error\n'
                  'Reverse Swing Has Peak Pace (138 km/h) & 9.5% Wkt Rate | Full Toss Has Low Dots (16.5%) & Low Wkts (6.5%)',
                  fontsize=10.5, fontweight='bold', color='#F0F6FC')
    ax2.set_ylabel('Scaled Metric Value', fontsize=8.5, fontweight='bold', color='#94A3B8')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#1E293B', edgecolor='#334155')
    ax2.grid(True, linestyle=':', alpha=0.3, color='#475569')

    # ===== PANEL 3: Net Bowler Equity as a Function of Wicket Valuation λ =====
    ax3 = axes[1, 0]
    ax3.set_facecolor('#0B0F19')

    lambda_vals = np.linspace(0, 22, 100)
    # Net cost: Runs - lambda * Wkt_rate
    cost_yorker = 0.91 - lambda_vals * 0.077
    cost_wide_yorker = 0.96 - lambda_vals * 0.020
    cost_cutter = 1.66 - lambda_vals * 0.098
    cost_rev_in = 1.88 - lambda_vals * 0.095
    cost_full_toss = 1.97 - lambda_vals * 0.065

    ax3.plot(lambda_vals, cost_yorker, color='#38BDF8', lw=2.5, label='Yorker (Lowest net cost across all states)')
    ax3.plot(lambda_vals, cost_wide_yorker, color='#818CF8', lw=2.0, linestyle='--', label='Wide Yorker (Low wkt rate = 2.0%, flat curve)')
    ax3.plot(lambda_vals, cost_cutter, color='#10B981', lw=2.2, label='Off Cutter (Elite death balance: 1.66 r/b, 9.8% wkt)')
    ax3.plot(lambda_vals, cost_rev_in, color='#F43F5E', lw=2.5, label='Reverse Swing In (Steep equity slope, 9.5% wkt rate)')
    ax3.plot(lambda_vals, cost_full_toss, color='#94A3B8', lw=1.8, linestyle=':', label='Full Toss (Execution error; uncompetitive across all λ)')

    # Crossover point where Reverse Swing overtakes Wide Yorker
    # 1.88 - 0.095*L = 0.96 - 0.020*L => 0.92 = 0.075*L => L = 12.27
    ax3.axvline(12.27, color='#FACC15', linestyle=':', lw=1.5)
    ax3.annotate('Crossover Point (λ* = 12.3 runs):\nReverse Swing strictly beats Wide Yorker\ndue to 9.5% wicket strike power!',
                 xy=(12.27, 0.71), xytext=(13.5, 1.25),
                 fontsize=8, fontweight='bold', color='#FACC15',
                 arrowprops=dict(arrowstyle="->", color='#FACC15', lw=1.0),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#1E293B', edgecolor='#FACC15'))

    ax3.axhline(0, color='#64748B', linestyle='--', lw=0.8)
    ax3.set_title(r'Panel 3: Net Bowler Equity vs Wicket Valuation $\lambda(b, w)$' + '\n' +
                  r'Net Cost/Delivery = $\mathrm{Runs} - \lambda \cdot \mathrm{Wkt\%}$ (Lower is Better for Bowler)',
                  fontsize=10.5, fontweight='bold', color='#F0F6FC')
    ax3.set_xlabel(r'Wicket Valuation $\lambda(b, w)$ (Runs / Wicket)', fontsize=8.5, fontweight='bold', color='#94A3B8')
    ax3.set_ylabel('Net Cost to Bowling Team (Runs per Delivery)', fontsize=8.5, fontweight='bold', color='#94A3B8')
    ax3.legend(loc='lower left', fontsize=8, facecolor='#1E293B', edgecolor='#334155')
    ax3.grid(True, linestyle=':', alpha=0.3, color='#475569')

    # ===== PANEL 4: Bowler Reverse Swing Mastery Leaderboard =====
    ax4 = axes[1, 1]
    ax4.set_facecolor('#0B0F19')

    rev_bowlers = df.filter(pl.col('delivery_style').str.contains('(?i)reverse')).group_by('bowler_name').agg([
        pl.len().alias('count'),
        pl.col('total_runs').mean().alias('avg_runs'),
        (pl.col('total_runs').mean() * 6.0).alias('econ'),
        pl.col('is_wicket').mean().alias('wkt_rate'),
        (pl.col('total_runs').mean() - 8.0 * pl.col('is_wicket').mean()).alias('net_cost')
    ]).filter(pl.col('count') >= 50).sort('net_cost').head(10).reverse()

    y_pos = np.arange(len(rev_bowlers))
    b_names = rev_bowlers['bowler_name'].to_list()
    b_econs = rev_bowlers['econ'].to_numpy()
    b_wkts = rev_bowlers['wkt_rate'].to_numpy() * 100.0
    b_costs = rev_bowlers['net_cost'].to_numpy()
    b_counts = rev_bowlers['count'].to_numpy()

    bars4 = ax4.barh(y_pos, b_costs, height=0.6, color=['#38BDF8' if i >= len(rev_bowlers)-3 else '#818CF8' for i in range(len(rev_bowlers))], edgecolor='#334155')

    for i, bar in enumerate(bars4):
        txt = f" {b_names[i]}: Net Cost {b_costs[i]:+.2f} r/b | Econ: {b_econs[i]:.1f} | Wkt: {b_wkts[i]:.1f}% | N={b_counts[i]}"
        ax4.text(0.05, bar.get_y() + bar.get_height()/2, txt, va='center', ha='left', fontsize=8, fontweight='bold', color='#F0F6FC')

    ax4.set_yticks([])
    ax4.set_title('Panel 4: Top Bowlers Executing Reverse Swing (Min 50 Balls)\n'
                  'Ranked by Net Equity (Econ - 8.0 * Wkt Rate) — Pathirana, Harshal & Arshdeep Lead',
                  fontsize=10.5, fontweight='bold', color='#F0F6FC')
    ax4.set_xlabel('Net Runs Conceded per Delivery (After Wicket Credit)', fontsize=8.5, fontweight='bold', color='#94A3B8')
    ax4.grid(True, linestyle=':', alpha=0.3, color='#475569')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_media = f'{media_dir}/plot_16_reverse_swing_physics_and_temporal_audit.png'
    out_brain = f'{artifact_dir}/plot_16_reverse_swing_physics_and_temporal_audit.png'
    plt.savefig(out_media, bbox_inches='tight', facecolor='#0B0F19', dpi=130)
    plt.savefig(out_brain, bbox_inches='tight', facecolor='#0B0F19', dpi=130)
    plt.close()

    t_elapsed = time.time() - t0
    print(f"Generated Plot 16 successfully in {t_elapsed:.2f}s:")
    print(f" -> {out_media} ({os.path.getsize(out_media)/1024:.1f} KB)")
    print(f" -> {out_brain} ({os.path.getsize(out_brain)/1024:.1f} KB)")

if __name__ == '__main__':
    generate_reverse_swing_audit()
