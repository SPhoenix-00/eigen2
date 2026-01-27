# Maverick Mode Logic Update

## Motivation
Previous runs showed that profitable agents were receiving negative fitness scores because the "hurdle rate" was being subtracted from all gains. This meant a trade making +0.5% profit (which is good) was scored as a loss if the hurdle was 0.6%. This misalignment caused the "Best Agent" tracking to stall and made it difficult to assess true performance.

## Changes

### 1. Reward Function Overhaul (Maverick Mode Only)
Moved from a **Subtractive (Punishment)** model to an **Additive (Bonus)** model.

*   **Dead Zone (-1.0% to +1.0%):**
    *   **Logic:** Linear reward (`Reward = Gain%`).
    *   **Effect:** A trade making +0.5% now yields **+0.5 points** (previously -0.7). A trade losing -0.5% yields **-0.5 points**.
    *   **Goal:** Ensure profitable agents always have positive fitness scores.

*   **Upside Bonus (> +1.0%):**
    *   **Logic:** `Reward = Gain + (Excess × 0.5)`.
    *   **Example:** A +2.0% gain yields `2.0 + (1.0 × 0.5) = 2.5` points.
    *   **Goal:** Incentivize "Maverick" behavior (sniping large moves) without spiking the signal too hard.

*   **Downside Penalty (< -1.0%):**
    *   **Logic:** `Reward = Loss - (Excess × 0.5)`.
    *   **Example:** A -2.0% loss yields `-2.0 - (1.0 × 0.5) = -2.5` points.
    *   **Goal:** Symmetrically punish large drawdowns to prevent reckless gambling.

### 2. Configuration Updates
*   Increased `LOCAL_BUFFER_SIZE` from 120,000 to **180,000** to allow for a larger replay history during local training runs.

### 3. Isolation
These changes are strictly isolated to `maverick_mode=True`. Normal and Consistency modes retain the original "hurdle as cost of doing business" logic to preserve their conservative character.
