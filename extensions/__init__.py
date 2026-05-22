"""Improvement methods for LLM JSON tool-output processing.

Three independent, plug-in techniques (no fine-tuning required):

* ``pruning``         -- HeuristicPlus context-aware JSON pruning (shrinks the input).
* ``plan_solve``      -- Plan-and-Solve with code generation (structures the reasoning).
* ``self_correction`` -- Self-correction with execution feedback (recovers from errors).
"""
