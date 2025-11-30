Gauntlet Mode: Solving the "Ghost Score" Problem

Executive Summary

We are shifting our training objective from "Run for N Generations" to "Achieve N Confirmed Breakthroughs." This change addresses a critical stability issue observed in Eigen2: the tendency for "breakthrough" agents to lose their performance edge immediately upon training resumption. We have diagnosed this not as a loss of learning, but as a failure of validation.

1. The Diagnosis: 
The "Ghost Score" TrapThe Observed PhenomenonIn current runs, we often see a pattern like this:
- Generation 40: Best Agent hits a fitness of 552.56.
- The Lock-in: The system records 552.56 as the "High Water Mark." Future agents must beat this score to be saved.
- The Resume: We stop and resume training. The system re-evaluates that exact same agent on new data slices.
- The Crash: The score drops to -1608.93.

The Root Cause
The score of 552.56 was likely a statistical illusion—a "Ghost Score." The agent got lucky on the specific random days selected for validation in Gen 40.Because the system locked in 552 as the baseline, the training loop is now trapped. It is trying to optimize agents to beat a "lucky ghost" rather than building robust general intelligence. Since the true performance is -1608, no agent will ever beat 552 honestly, leading to stalled learning and "disheartening" drops.

2. The Solution: Gauntlet Mode

We introduce a new state machine governing the training loop. Instead of blindly trusting high scores, we treat them as hypotheses that must be rigorously tested.

The Protocol
- Step 1: Detection (The Spark)
    We monitor the population for any agent that exceeds our Confirmed Baseline (initially 0) by a significant margin (e.g., 10%).
    Status: "Potential Breakthrough" detected.
- Step 2: Stabilization (The Siege)
    We do not celebrate yet. We lock the training focus on this candidate for 3 generations.
    Goal: Allow the Actor and Critic networks to converge around this new behavior, preventing "one-hit wonders" that vanish in the next gradient step.
- Step 3: The Gauntlet (The Test)
    Once stabilized, the candidate faces The Gauntlet:
    - Standard Validation: 7 random data slices.
    - The Gauntlet: 20 rigorously randomized slices covering different market regimes across all training and validation data
    - Scoring: We use a "Slightly Forgiving Aggregator" (0.75*mean + 0.25*min) to balance robustness with average performance
- Step 4: The Ratchet (The Fix)This is the most critical change.
    FIRST BREAKTHROUGH: We "give away" the first breakthrough - any gauntlet score (even negative) becomes the baseline.
    This provides a realistic starting point since we don't know in advance what performance level is achievable.

    SUBSEQUENT BREAKTHROUGHS: If a candidate scores 50.0 in the Gauntlet:
    - It is lower than the lucky spike of 552.
    - But it is higher than the previous Confirmed Baseline (e.g., -100).

    Action: We CONFIRM the breakthrough.The Ratchet: We set the new "High Water Mark" to 50.0 (the reality), NOT 552 (the ghost).
    
3. Why This Works

By resetting the bar to the Gauntlet Score, we align the training objective with reality.
Objective:
    Old Logic: Beat the highest score ever seen (even if lucky).
    Gauntlet Logic: Beat the highest robust score confirmed.
Resuming:
    Old Logic: Massive drop in score ("Reality Check").
    Gauntlet Logic: Stable score (Reality was already checked).
Progress:
    Old Logic: Linear generations (often stalling).
    Gauntlet Logic: Stair-step "Breakthroughs" (ratcheting up).
Outcome:
    Old Logic: An agent that got lucky once.
    Gauntlet Logic: An agent that survived a stress test.
    
4. The New Metric: "Breakthrough Velocity"

Instead of asking "How many generations did we run?", we will measure progress by "Time to Next Breakthrough."

Run Goal: Achieve 4 Confirmed Breakthroughs (to be confirmed in config.py)