"""
cost_aware.py — Cost-aware neural routing with Pareto frontier optimization.

THIS IS PHASE 5 (hours 12-20). Don't run this until Phase 3 is complete.

The key idea: instead of just minimizing classification error, we add a
cost penalty that pushes the model toward cheaper retrieval paradigms when
it's uncertain about the query type.

Loss = L_classification + λ * L_cost

where L_cost = expected token cost of the predicted paradigm:
    L_cost = Σ_i softmax(logit_i) * cost(paradigm_i)

By sweeping λ from 0 (pure accuracy) to large values (pure cost savings),
we trace a Pareto frontier of cost vs. quality.

This is the main novel contribution of the paper.

Usage (after Phase 3):
    python src/cost_aware.py --lambda_cost 0.1 --model distilbert-base-uncased
    python src/cost_aware.py --sweep  # sweeps λ from 0 to 1.0
"""

# Implementation will be added in Phase 5.
# See EXECUTION_PLAN.md for details.

raise NotImplementedError(
    "This module is Phase 5 (hours 12-20). "
    "Complete Phases 1-3 first, then we'll implement this together."
)
