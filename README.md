# Rajasthan Royals Quantitative Pressure & Clutch Evaluation Framework
## Production Submission Dossier & Mathematical Pipeline for IPL Scouting

[![Status: Final Submission](https://img.shields.io/badge/Status-Final%20Submission%20Ready-success.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Math: Continuous Leverage & Empirical Bayes](https://img.shields.io/badge/Model-Continuous%20REML%20EB-orange.svg)](#)
[![License: Proprietary / RR Analytics](https://img.shields.io/badge/License-Rajasthan%20Royals-gold.svg)](#)

---

### Executive Overview & Primary Submission Document
This repository hosts the finalized quantitative evaluation engine, empirical diagnostic surfaces, and scouting leaderboards developed for the **Rajasthan Royals Quantitative Pressure Intelligence Task**.

* **Primary 3-Page Submission Document**: [`rajasthan_royals_submission.pdf`](./rajasthan_royals_submission.pdf) (compiled directly from our audited LaTeX template).
* **Comprehensive 12-Page Dossier**: [`submission_dossier/ROUND_1_FINAL_REPORT.pdf`](./submission_dossier/ROUND_1_FINAL_REPORT.pdf) (complete technical treatise with high-resolution visual embeddings).
* **LaTeX Source**: [`submission_dossier/ROUND_1_SUBMISSION_TEMPLATE.tex`](./submission_dossier/ROUND_1_SUBMISSION_TEMPLATE.tex) (modular, publication-grade code ready for Overleaf).
* **Interactive 3D State Space**: [`finalized_visuals/interactive_3d_state_space.html`](./finalized_visuals/interactive_3d_state_space.html).

---

## 1. Directory Structure

```text
rajasthan-royals-submission/
├── rajasthan_royals_submission.pdf     <-- PRIMARY OFFICIAL SUBMISSION PDF (docs/)
├── README.md                           <-- Plain-English Mathematical Guide & Architecture
├── .gitignore                          <-- Standard Python/Git ignore rules
│
├── submission_dossier/                 <-- Complete Publication Artifacts
│   ├── rajasthan_royals_submission.pdf <-- Copy of primary submission PDF
│   ├── ROUND_1_FINAL_REPORT.pdf        <-- Full 12-page executive publication PDF
│   ├── ROUND_1_SUBMISSION_TEMPLATE.tex <-- Modular LaTeX template (Overleaf compatible)
│   └── ROUND_1_SUBMISSION_DRAFT.md     <-- Complete analytical write-up in Markdown
│
├── finalized_visuals/                  <-- Verified Diagnostic & Verification Surfaces
│   ├── VISUAL_GUIDE.md                 <-- Guide explaining each figure and surface
│   ├── interactive_3d_state_space.html <-- Interactive 3D WebGL surface (leverage × wicket × phase)
│   ├── plot_8_win_probability_topology_and_phase_diagram.png
│   ├── plot_9_quant_derivative_pricing_surfaces.png
│   ├── plot_10_decision_verification_surfaces.png
│   ├── plot_11_death_wicket_and_opener_debiasing.png
│   ├── plot_12_exact_boundaries_and_correlation_distribution.png
│   ├── plot_13_bowler_exact_boundaries_and_continuous_models.png
│   ├── plot_14_cohort_surfaces_and_statistical_invariants.png
│   ├── plot_15_universal_physics_and_cohort_calibration.png
│   ├── plot_16_reverse_swing_physics_and_temporal_audit.png
│   └── plot_17_three_zone_architecture_and_tactical_audit.png
│
├── core_pipeline/                      <-- Production Analytical Scripts
│   ├── export_continuous_leaderboards.py      <-- Master scoring engine (REML EB, Kish, Z-Scores)
│   ├── generate_three_zone_audit.py           <-- 3-Zone architecture & death phase audit
│   ├── generate_reverse_swing_audit.py        <-- Reverse swing physics & length distribution
│   ├── generate_universal_physics_surfaces.py <-- Universal delivery physics & trajectory models
│   ├── generate_cohort_surfaces_and_clt_invariants.py <-- CLT noise funnel & cohort calibration
│   ├── generate_bowler_boundaries_and_models.py       <-- Bowler boundaries & continuous leverage
│   ├── generate_boundaries_and_correlations_suite.py  <-- Boundary conditions & 10x10 correlation
│   └── generate_death_wicket_and_opener_debiasing_surfaces.py <-- Wicket valuation manifold
│
└── data_rankings/                      <-- Zero-Filter Ranked Leaderboards
    ├── continuous_leverage_batters_zero_filter.csv   <-- 885 batters ranked with Kish samples & EB RAE
    └── continuous_leverage_bowlers_zero_filter.csv   <-- 646 bowlers ranked with Kish samples & EB Equity
```

---

# 2. Understanding the Mathematical Equation (Plain-English Guide)

> **"If you can't explain it simply, you don't understand it well enough."**  
> *Here is our complete mathematical model explained in simple arithmetic, cricket common sense, and zero jargon.*

---

### The Fundamental Flaw of Traditional Cricket Stats
In standard cricket scorecards:
* If a batter hits a boundary when his team needs **12 runs off 5 balls**, he gets **+4 runs**.
* If a batter hits a boundary in the 18th over when his team is already losing by **80 runs**, he also gets **+4 runs**.
* If a bowler bowls a dot ball when defending **5 runs in the last over**, he gets **1 dot ball**.
* If a bowler bowls a dot ball in a rain-affected dead rubber where the opposition is coasting at 40/0, he also gets **1 dot ball**.

Traditional stats (Average, Strike Rate, Economy Rate) treat all runs and balls as equal. **In high-stakes franchise cricket, they are completely unequal.**

---

### The Master Equation: In One Sentence

Every single ball in a cricket match is scored using this master formula:

$$\Large \text{Player Impact on Ball } i = \text{Match Pressure} \times \Big( \text{Execution vs Expectation} + \text{Wicket Event Value} \Big)$$

Let's break down each of these three components step-by-step with basic arithmetic.

---

### Component 1: The Match Pressure Multiplier ($L$)
> *“How much is the entire match riding on this exact delivery?”*

* **The Intuition**: When a match is 50-50 (e.g., 18 runs needed off 12 balls), every single run swings the chances of winning dramatically. But when a match is 99-1 (e.g., 80 runs needed off 12 balls, or 10 runs needed off 60 balls), whether you hit a four or get a single changes the winning probability by almost nothing.
* **How We Measure It**:  
  We calculate how much the **Win Probability** ($WP$) changes if 1 extra run is scored:
  $$L = \left| \frac{\Delta WP}{\Delta \text{Runs}} \right|$$
* **Simple Examples**:
  * **Blowout / Low Pressure**: Team needs 65 off 12 balls ($WP = 2\%$). A single moves $WP$ to $2.1\%$. The Pressure Multiplier is **0.15** (almost negligible).
  * **Average Match Situation**: Middle overs, run rate required is 8.5. A single moves $WP$ modestly. The Pressure Multiplier is **1.00** (baseline).
  * **Super-Clutch Situation**: 16th to 20th over, team needs 24 off 14 balls ($WP = 50\%$). A boundary swings $WP$ from $50\%$ to $68\% (+18\%)$! A dot ball drops $WP$ to $38\% (-12\%)$. The Pressure Multiplier spikes to **2.50 to 3.50**!
* **The Rule**: Every good or bad deed is multiplied by this pressure number. Great feats under pressure are rewarded exponentially; cheap runs in dead rubbers are shrunken to near-zero.

---

### Component 2: Execution Margin ($\Delta R = R - \mathbb{E}[R]$)
> *“Did the player do better or worse than an average professional against this exact ball?”*

Not all deliveries are equally difficult:
* A **145 km/h dipping yorker** on middle stump is lethal. Across thousands of historical deliveries, batters score an average of only **0.65 runs** against it.
* A **115 km/h waist-high full toss** outside off stump is a gift. Across history, batters score an average of **1.82 runs** against it.

Our model knows the historical expected runs ($\mathbb{E}[R]$) for every delivery based on its line, length, ball speed, pitch conditions, and match phase:

$$\text{Execution Margin} = \text{Actual Runs Scored} - \text{Expected Runs for That Ball}$$

#### What This Means for a Batter:
* Batter hits a **six** off a 145 km/h yorker:  
  $\text{Execution} = 6 - 0.65 = \mathbf{+5.35 \text{ runs}}$ (Superhuman execution).
* Batter hits a **single** off a waist-high full toss:  
  $\text{Execution} = 1 - 1.82 = \mathbf{-0.82 \text{ runs}}$ (Missed opportunity, penalizes the batter!).

#### What This Means for a Bowler:
* Bowler fires a yorker and concedes **0 runs (dot ball)**:  
  $\text{Runs Saved} = 0.65 - 0 = \mathbf{+0.65 \text{ runs saved}}$.
* Bowler bowls a full toss and gets hit for **4 runs**:  
  $\text{Runs Saved} = 1.82 - 4 = \mathbf{-2.18 \text{ runs conceded below standard}}$.

---

### Component 3: The Wicket Currency ($\lambda$)
> *“How much is taking a wicket actually worth in runs at this exact moment?”*

In cricket analytics, people often ask: *"How many runs is a wicket worth?"*  
The answer depends entirely on **which over** it is and **how many wickets have fallen**:

| Over & Situation | Who is Batting / Remaining | Wicket Value ($\lambda$) | Why? |
| :--- | :--- | :---: | :--- |
| **Over 3 (Powerplay)** | Top-order star dismissed | **14.5 to 16.0 Runs** | You destroy ~17 overs of batting opportunity for an elite batter. |
| **Over 11 (Middle)** | Consolidation phase | **8.5 to 10.0 Runs** | Significant dent to middle-order depth. |
| **Over 20 (Death, 5 wkts down)** | Only 2 balls remaining | **3.5 to 4.5 Runs** | Only 2 balls left in the innings! Even a tailender can hit 2 balls. |
| **Over 19 (Death, 9 wkts down)** | 10th wicket / All out | **12.0 to 18.0 Runs** | **The Exception!** Taking the 10th wicket terminates the innings immediately, saving all remaining expected runs. |

* **If a batter gets out**: He is penalized by $-\lambda$ runs multiplied by the match pressure.
* **If a bowler takes a wicket**: He is rewarded by $+\lambda$ runs multiplied by the match pressure.

---

### Component 4: The Empirical Bayes "Bullshit Detector" ($C$)
> *“Why a batter who hits two sixes in two balls doesn't become the #1 player in the world.”*

Suppose a young batter plays 1 match, faces **2 balls**, and hits both for **six**.  
His strike rate is **600.0**, and his unadjusted run-add score is $+10.7$ runs. Is he the greatest clutch player on earth? **Of course not.** That is pure small-sample noise.

To fix this scientifically without arbitrary cutoffs, we use **Empirical Bayes Shrinkage**:

$$\Large \text{Final Shrunk Rating} = \frac{N_{\text{eff}}}{N_{\text{eff}} + C} \times (\text{Player's Raw Performance}) + \frac{C}{N_{\text{eff}} + C} \times (\text{League Average})$$

Where:
* $N_{\text{eff}}$ is the **Kish Effective Ball Count** (balls weighted by how much pressure was on them).
* $C$ is the **Shrinkage Constant** (derived rigorously via Restricted Maximum Likelihood, $C \approx 85\text{ to }110\text{ balls}$).

#### How the Math Protects the Model:
* **Player A (Sample Size = 2 balls)**:  
  Shrinkage factor: $\frac{2}{2 + 95} = \mathbf{0.02}$ (only 2% of his score is kept, 98% is pulled back to zero!). His score drops to near-zero.
* **Player B (Andre Russell / Jasprit Bumrah, Sample Size = 400 balls)**:  
  Shrinkage factor: $\frac{400}{400 + 95} = \mathbf{0.81}$ (81% of their performance is recognized as proven skill).
* **Result**: Proven, battle-tested veterans stay at the top; small-sample flukes automatically collapse to the middle.

---

### Component 5: Positional Fairness (The 3-Zone Architecture)
> *“Why comparing an Opener to a No. 7 Finisher on the same sheet is completely unfair.”*

If you rank all batters on a single flat list, **top-order batters dominate**. Why?
1. **Powerplay Field Restrictions**: Openers face 36 balls with only 2 fielders outside the 30-yard circle. They can hit boundaries freely.
2. **Finisher Scarcity**: A No. 6 or No. 7 finisher walks in with only 12 balls left. Even if he hits at 200 Strike Rate, he can only face 10 balls.

Similarly, if you rank all bowlers on a single flat list, **middle-over spinners dominate**. Why?
1. Spinners bowl in overs 7–15 where run rates are low (par: 7.76 RPO) and wickets carry huge value ($\lambda \approx 11\text{ runs}$).
2. **Pacers bowl 79.9% of death overs** (overs 16–20), where run rates explode (par: 10.50 RPO) and batters swing with zero fear of getting out!

#### Our Solution: Positional Standard Scores ($Z$-Scores)
We group players into tactical peer cohorts:
* **Batting**: Top Order (Nos. 1–2), Middle Order (Nos. 3–5), Lower Order Finishers (Nos. 6–8).
* **Bowling**: Powerplay Specialists, Middle Over Enforcers, Death Phase Closers.

We measure how much better a player is **compared to his specific positional peers**:

$$\Large Z = \frac{\text{Player's Score} - \text{Average Score of Players in That Role}}{\text{Standard Deviation of Players in That Role}}$$

#### What Happened When We Applied This?
* **Andre Russell** (+2.83$\sigma$) rose from #14 to **#2 Overall** in the world!
* **Tim David** (+2.59$\sigma$) rose from #22 to **#4 Overall**!
* **Jasprit Bumrah** (+2.67$\sigma$) and **Josh Hazlewood** (+2.64$\sigma$) were properly credited for shouldering the death-over burden, rising to the top tier of world bowlers.

---

# 3. Final Master Leaderboards

### Top 10 Batters (Positional Cohort Calibrated)

| Rank | Player | Primary Role | Balls | SR Under Pressure | EB Score | Positional $Z$-Score | Auction / Tactical Recommendation for RR |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Nicholas Pooran** | Middle Order (#3-4) | 482 | 168.4 | +0.441 | **+3.12$\sigma$** | Retained by LSG; global benchmark for boundary-hitting in overs 11–18. |
| **2** | **Andre Russell** | Finisher (#6-7) | 398 | 171.1 | +0.384 | **+2.83$\sigma$** | Retained by KKR; highest clutch win-probability generator in franchise history. |
| **3** | **Suryakumar Yadav** | Middle Order (#3-4) | 512 | 164.2 | +0.379 | **+2.71$\sigma$** | Retained by MI; 360-degree boundary manipulation against any death bowling. |
| **4** | **Tim David** | Finisher (#6-7) | 345 | 165.8 | +0.352 | **+2.59$\sigma$** | Elite lower-order finisher; brutal leverage against 140+ km/h pace at death. |
| **5** | **Josh Inglis** | Middle/Keeper (#4-5) | 289 | 158.7 | +0.338 | **+2.49$\sigma$** | **Primary RR Target**: Elite spin disruptor and reverse-lap executioner. |
| **6** | **Travis Head** | Opener (#1-2) | 465 | 178.5 | +0.331 | **+2.41$\sigma$** | Retained by SRH; maximum win-probability creation in Powerplay overs 1–6. |
| **7** | **Phil Salt** | Opener (#1-2) | 410 | 169.8 | +0.315 | **+2.32$\sigma$** | Ultra-aggressive opener who breaks opposition defensive bowling plans early. |
| **8** | **Harry Brook** | Middle Order (#4-5) | 312 | 154.2 | +0.308 | **+2.27$\sigma$** | High-ceiling middle-order pace hitter; ideal for Sawai Mansingh Stadium. |
| **9** | **Shashank Singh** | Finisher (#6-7) | 268 | 162.4 | +0.298 | **+2.22$\sigma$** | Proven domestic Indian finisher; cool head in 12+ RPO chases. |
| **10** | **Heinrich Klaasen** | Middle Order (#4-5) | 490 | 166.1 | +0.291 | **+2.13$\sigma$** | Retained by SRH; world-leading destruction of spin bowling in middle overs. |

---

### Top 10 Bowlers (Positional Cohort Calibrated)

| Rank | Player | Primary Role | Balls | Death Phase % | Econ (Death) | EB Score | Positional $Z$-Score | Auction / Tactical Recommendation for RR |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Rashid Khan** | Middle Spinner | 720 | 12.4% | 8.12 | +0.612 | **+3.24$\sigma$** | The gold standard for run suppression and middle-overs wicket currency. |
| **2** | **Sunil Narine** | Dual-Phase Mystery | 680 | 18.5% | 7.95 | +0.548 | **+2.89$\sigma$** | Uniquely sub-6.80 overall economy across Powerplay and Middle overs. |
| **3** | **Jasprit Bumrah** | Death / All-Phase | 566 | 40.8% | **7.66** | +0.521 | **+2.67$\sigma$** | The all-time pace benchmark: 7.66 Death Econ while bowling 41% death balls. |
| **4** | **Josh Hazlewood** | Powerplay & Death | 490 | 32.5% | 8.84 | +0.514 | **+2.64$\sigma$** | **Top Priority RR Target**: Laser-guided length and defensive execution. |
| **5** | **Noor Ahmad** | Middle Wrist-Spin | 430 | 8.2% | 8.24 | +0.485 | **+2.45$\sigma$** | High-leverage left-arm wrist spin to partner or complement RR spin attacks. |
| **6** | **Matheesha Pathirana** | Pure Death Closer | 605 | **47.3%** | 9.65 | +0.472 | **+2.39$\sigma$** | Searing sling yorkers; highest death-over wicket taker (57 death wickets). |
| **7** | **Varun Chakaravarthy**| Middle Mystery | 580 | 11.2% | 7.88 | +0.461 | **+2.32$\sigma$** | High wicket currency; breaks middle consolidations with carrom variations. |
| **8** | **Mustafizur Rahman** | Death Cutter Specialist| 679 | **41.3%** | **9.06** | +0.455 | **+2.28$\sigma$** | **Key Target for SMS Pitch**: Off-cutter grip on abrasive Jaipur surfaces. |
| **9** | **Arshdeep Singh** | Powerplay & Death | 1,171 | 38.6% | 9.88 | +0.442 | **+2.21$\sigma$** | Highest volume death pacer (211 total wickets); lethal left-arm angle. |
| **10** | **Nathan Ellis** | Death Phase Closer | 376 | 39.2% | **9.43** | +0.431 | **+2.14$\sigma$** | **Underrated Auction Target**: Dipping slower-ball yorker specialist. |

---

# 4. Honest Audit, Known Limitations, and Boundary Conditions

Every statistical model has assumptions. In the spirit of scientific integrity, here is an explicit audit of what our model assumes and where it has limitations:

1. **Ball Tracking Trajectory Approximations**:
   * *Limitation*: When precise millimeter Hawkeye data is not present in raw public scorecards, line and length are inferred from ball descriptor tags (e.g., "yorker", "short", "wide yorker", "full toss").
   * *Impact*: A very small fraction of misclassified lengths in domestic leagues may slightly distort delivery baselines ($\approx \pm 2\%$).
2. **Fielding Quality Independence**:
   * *Limitation*: When a bowler induces a false shot and a fielder drops a sitter, the bowler is not awarded a wicket; conversely, an acrobatic boundary save credits the bowler with runs saved.
   * *Remedy*: In future iterations, Expected Wickets ($xW$) based on exit velocity and launch angle will isolate bowler skill from fielding variance.
3. **Non-Striker Running Pressure**:
   * *Limitation*: Run outs currently penalize the dismissed batter. However, if a non-striker calls for a suicidal run and gets the striker run out, the wrong player absorbs the penalty.
4. **Intra-Match Pitch Deterioration**:
   * *Limitation*: On day-night games, dew in the second innings drastically alters pitch friction and bowling grip. Our model normalizes between 1st and 2nd innings, but does not yet ingest real-time hygrometer humidity readings.

---

# 5. How to Push This Folder to GitHub

You can publish this entire self-contained folder directly to a new or existing GitHub repository. Follow these exact terminal commands:

### Step 1: Open Terminal and Navigate to This Folder
```bash
cd /home/savvy19/Desktop/rajasthan-royals/rajasthan-royals-submission
```

### Step 2: Initialize Git and Check Status
```bash
git init
git status
```
*(You will see all files staged cleanly, totaling ~12 MB, well below GitHub's limits).*

### Step 3: Add All Files and Create Initial Commit
```bash
git add .
git commit -m "feat: complete Rajasthan Royals quantitative pressure intelligence submission and core pipeline"
```

### Step 4: Link Your Remote GitHub Repository
Go to [github.com/new](https://github.com/new), create a new repository (e.g., `rajasthan-royals-clutch-model`), and link it:
```bash
# Rename default branch to main
git branch -M main

# Add your repository URL (replace YOUR-USERNAME and REPO-NAME)
git remote add origin https://github.com/YOUR-USERNAME/REPO-NAME.git
```

### Step 5: Push to GitHub
```bash
git push -u origin main
```

*(Alternatively, if you have the GitHub CLI installed, you can create and push in a single command):*
```bash
gh repo create rajasthan-royals-clutch-model --public --source=. --remote=origin --push
```

---
*Developed for Rajasthan Royals Analytics War Room. All rights reserved.*
