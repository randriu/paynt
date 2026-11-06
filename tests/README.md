# PAYNT tests

How to run them is in the main README, section "Testing PAYNT". `pytest tests payntbind/tests` runs both suites in one process.

## Layout

Most folders mirror `paynt/`: `dt/`, `mdp_family/`, `model/`, `parameter_space/`, `parser/`, `pomdp/` (also Dec-POMDPs and POSMGs), `specification/`, `synthesizer/`, `utils/`, and the top-level `test_*.py` files for the top-level modules. The others:

| folder | contents |
|---|---|
| `coloring/` | colorings: `Coloring`, `ColoringGeneral`, `ColoringBuilder`, custom and incomplete colorings, choice formulas |
| `cli/` | the command line: in this process (`test_cli.py`, with the exports read back), or in a process of its own, so that a crash or an abort at exit fails one test |
| `soundness/` | every engine against the reference answers, on tiny and on random sketches; known wrong answers are expected failures, so that a fix shows up |
| `helpers/` | `models.py` loads the models of `models/`, `reference.py` model-checks every design of a tiny sketch on its own, `engines.py` runs every engine and compares it with that, `formulas.py` makes random coloring formulas |
| `performance/` | the weekly benchmarks, not run by pytest |

## Loading models

Every model of `models/tests` that several tests share has a fixture in `conftest.py`, named after its folder (`smpmc_tiny`, `dt_orchard`, ...). The fixture is parsed once per test and has `.factory`, `.task` and `.colored_mdp`.

Other models are loaded with `tests.helpers.models`: `load_sketch(project)` gives `(factory, task)`, `load_colored_mdp(project)` gives `(colored_mdp, task)`. A synthesizer changes its task, so a test that runs two synthesizers loads the model twice.

## Test models

The models in `models/tests`. "Designs" is the size of the parameter space.

| model | kind | states / choices | parameters / designs | properties | used for |
|---|---|---|---|---|---|
| `smpmc-tiny` | DTMC sketch | 8 / 15 | 3 / 8 | `P>=0.1 [F "done"]`; `optimality.props`: `Pmax`; `sketch.costs` for the costs constraint | SMPMC, constraints, robust synthesis |
| `generic-maze` | DTMC sketch | 183 / 257 | 24 / 9.4e6 | `R{"steps"}min` | model index, scoring, SMPMC theory (an ill-conditioned DTMC) |
| `time-energy` | DTMC sketch; `tree.templ`: MDP | 3 / 7; 3 / 9 | 2 / 9; 1 / 5 (depth 0) | time and energy constraints: `sketch.props` feasible, `infeasible.props` not; `optimality.props`: least energy within 4 hours; `tree.props` for trees | multi-constraint splitting, a known AR bug |
| `objective-optimum-violates-constraint` | DTMC sketch | 3 / 4 | 1 / 2 | `P>=0.9` and `R{"cost"}min` | known AR bug |
| `max-reward-infinite-designs` | DTMC sketch | 4 / 5 | 1 / 2 | `R{"cost"}max`; `lower.props`: `R>=2.5` | known AR and SMPMC bug |
| `constraint-decided-by-one-design` | DTMC sketch (`hybrid.templ`) | 4 / 8 | 1 / 3 | a probability and a cost constraint | known AR and hybrid bug |
| `edge-cases` | DTMC sketches | 3 / 3-6 | 1-2 / 1-4 | none: the properties are in `soundness/test_engines.py` | thresholds equal to a value, goals reached at once or never, degenerate holes, a deadlock |
| `prob-goal` | DTMC sketch; `family.templ`: MDP family | 4 / 5; 3 / 6 | 1 / 2; 1 / 3 | `P>=0 [F "goal"]` | the prob0/prob1 constraint |
| `mdp-family-avoid-8-2-easy` | MDP family | 21169 / 89776 | 6 / 3584 | `P>0.9` | policy trees, families in SMPMC, the coloring builder |
| `mdp-family-rocks-4-2` | MDP family | 162 / 594 | 4 / 81 | `P>=1`; `max.props`: `Pmax` | robust SMPMC |
| `mdp-family-correlated` | MDP family | 4 / 8 | 1 / 2 | `Pmax`; `min.props`: `Pmin` | robust optimality |
| `incomplete-two-targets` | MDP family, incomplete coloring | 3 / 4 | 1 / 2 | two `P>=1` constraints | incomplete colorings |
| `pomdp-family-avoid-smaller` | POMDP family | 14941 / 35616 | 6 / 64 | `R{"rewardmodel_penalty"}min` | POMDP families |
| `pomdp-maze` | POMDP | 15 / 57 | 7 / 16384 (memory 1) | `R{"steps"}min` | FSC synthesis |
| `decpomdp-dectiger` | Dec-POMDP, Cassandra format | 11 / 91 | 6 / 729 | none: the discounted reward of the model | Dec-POMDPs |
| `posmg-mec-test` | POSMG | 4 / 6 | 1 / 3 | `Pmax` | POSMG basics, no memory unfolding |
| `posmg-test-game` | POSMG | 8 / 14 | 3 / 8 | `Pmax` | memory unfolding |
| `dt-maze` | MDP | 15 / 67 | 1 / 7 | `R{"steps"}min` | decision trees, choice formulas |
| `dt-orchard` | MDP | 6869 / 19349 | 1 / 7 | `Pmax` | decision trees, dtnest |

Where the answers the tests expect come from:
- by hand:
  - `objective-optimum-violates-constraint`: cost 2
  - `max-reward-infinite-designs`: 3.0 at EXIT=1
  - `mdp-family-correlated`: worst cases 0.45 and 0.36, worked out in its sketch
  - `smpmc-tiny`: `Pmax` 1.0, with x1=4
  - `time-energy`: the trips of 4 hours, (0,2), (1,1) and (2,0); `tree.props` needs a tree of depth 1; `optimality.props`: 4 units of energy
  - `constraint-decided-by-one-design` and `edge-cases`: in the comments of the sketches
- all designs checked one by one: `tests/helpers/reference.py`, with Storm in exact arithmetic; `soundness/test_reference.py` checks it against the answers by hand
- another engine: the decision-tree optima, where AR and SMPMC agree
- solver output, so a regression check only:
  - `pomdp-maze`: 71.93
  - `decpomdp-dectiger`: -19.9987
  - `posmg-test-game`: 0.375 and 0.4395
  - `generic-maze`: 9.04e8, the worst case of an ill-conditioned sub-MDP
  - `dt-orchard`: 0.4845

## Writing tests

- Check a result against something independent of the code under test: a value worked out by hand, all designs checked one by one (`tests/helpers/reference.py`), another engine, or a model check of the returned assignment.
- Say where an expected number comes from. A number copied from PAYNT's output is marked as a regression value.
- The name of a test says what it asserts.
- Docstrings and comments explain the test, not its history.
- Seeds are fixed.
- Tolerances follow from the model checking precision. Exact comparisons only for exact values.
- A timeout or an unknown answer never counts as infeasible.
- Match error messages only for errors a user sees.
- A test takes at most a minute or two. Larger runs belong in `performance/`.

## Random sketches

`soundness/test_random_sketches.py` runs every engine on 25 random sketches. More of them look for new bugs, e.g. `PAYNT_RANDOM_SKETCHES=1000 pytest tests/soundness/test_random_sketches.py -n auto` (3 minutes on 8 cores); `PAYNT_RANDOM_FIRST_SEED` picks other sketches. The stable and latest CI run 500 new ones every time, starting from a seed given by the run number. A failure prints the sketch and its properties, and its test id is its seed; once reduced by hand, it becomes a model of `models/tests` with a case in `soundness/test_engines.py`.
