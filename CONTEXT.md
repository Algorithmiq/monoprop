# Monoprop

Terminology for retained operators and their numerical comparison.

## Language

**Propagator instance**:
An independent propagation context for an operator and its evolution. A copy is a separate instance with independent
operator state.
_Avoid_: Partition when referring to a copied propagator

**Rank-local operator**:
The portion of one propagator instance's global retained operator assigned to one rank.
_Avoid_: The process's operator when several independent propagator instances are present

**Global retained operator**:
The collection of retained term keys and associated coefficients for one propagated operator, independent of how those
terms are distributed.
_Avoid_: Energy or gradient when referring to the operator itself

**Retained-operator equivalence**:
Identical global retained term keys, with corresponding coefficients equal within the applicable numerical tolerances.
Agreement of energies or gradients alone does not establish retained-operator equivalence.
_Avoid_: Observable agreement as a synonym
