# PAYNT

[![CI (Latest)](https://github.com/randriu/paynt/actions/workflows/ci_latest.yml/badge.svg?branch=master)](https://github.com/randriu/paynt/actions/workflows/ci_latest.yml)
[![CI (Stable & Fetched)](https://github.com/randriu/paynt/actions/workflows/ci_stable.yml/badge.svg?branch=master)](https://github.com/randriu/paynt/actions/workflows/ci_stable.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/paynt)](https://pypi.org/project/paynt/)

PAYNT (Probabilistic progrAm sYNThesizer) is a tool for the automated synthesis of probabilistic programs. PAYNT takes a program with holes (a so-called sketch) and a PCTL specification, and outputs a concrete hole assignment that yields a satisfying program, if such an assignment exists. PAYNT also supports the synthesis of finite-state controllers for POMDPs, Dec-POMDPs and one-sided POSMGs, synthesis of decision trees for MDPs and synthesis of policy trees for families of MDPs. Internally, PAYNT interprets the incomplete probabilistic program as a family of Markov chains and uses state-of-the-art synthesis methods on top of the model checker [Storm](https://github.com/moves-rwth/storm) to identify satisfying realization. PAYNT is implemented in Python and uses [stormpy](https://github.com/moves-rwth/stormpy), Python bindings for Storm. PAYNT is hosted on [github](https://github.com/randriu/paynt).

PAYNT is described in 
- [1] PAYNT: A Tool for Inductive Synthesis of Probabilistic Programs by Roman Andriushchenko, Milan Ceska, Sebastian Junges, Joost-Pieter Katoen and Simon Stupinsky. In: CAV'21.
- [2] An Oracle-Guided Approach to Constrained Policy Synthesis Under Uncertainty by Roman Andriushchenko, Milan Ceska, Sebastian Junges, Joost-Pieter Katoen and Filip Macak. Journal of Artificial Intelligence Research (2025).


## Installation

### (a) For users

To download and install PAYNT, use:

```shell
pip install paynt
```

Alternatively, you may build PAYNT from source:

```shell
git clone https://github.com/randriu/paynt.git
cd paynt
python3 -m venv venv && source venv/bin/activate
pip install .
```

### (b) For developers

PAYNT depends on [Storm](https://github.com/moves-rwth/storm) and [stormpy](https://github.com/moves-rwth/stormpy). For developers, we recommend having local installations of both Storm and stormpy (see [section below](#installing-storm-and-stormpy)). If you have stormpy installed in your developer environment, you can use:

```shell
pip install -r build-requirements.txt
pip install . --no-build-isolation
```

which builds and installs PAYNT directly into your environment. **Note that the Storm backends used by both PAYNT and stormpy need to be the same.** While we implemented several routines that check the backend compatibility, it is up to the developer to make sure of it.

PAYNT is also available as a docker image:

```shell
docker pull randriu/paynt
docker run --rm -it randriu/paynt
python3 -m paynt --help
```

#### Installing Storm and stormpy

Please refer to [Storm documentation](https://www.stormchecker.org/documentation/obtain-storm/build.html) and [stormpy documentation](https://moves-rwth.github.io/stormpy/installation.html) for more information. Here we provide a list of commands that build master branch of Storm and stormpy in virtual environment without further explanation:

```shell
python3 -m venv venv && source venv/bin/activate
mkdir prerequisites && cd prerequisites
git clone https://github.com/moves-rwth/storm.git
git clone https://github.com/moves-rwth/stormpy.git
mkdir storm/build && cd storm/build
cmake ..
make storm storm-cli storm-pomdp
cd - && cd stormpy
pip install . --config-settings=cmake.define.STORMPY_USE_STORM_DFT=OFF --config-settings=cmake.define.STORMPY_USE_STORM_GSPN=OFF
```

## Running PAYNT

PAYNT can be executed using the command in the following form:

```shell
python3 -m paynt PROJECT [OPTIONS]
```
where ``PROJECT`` is the path to the benchmark folder and the most important options are:
- ``--sketch SKETCH``: the file in the ``PROJECT`` folder containing the template description or a POMDP program [default: ``sketch.templ``]
- ``--props PROPS``: the file in the ``PROJECT`` folder containing synthesis specification [default: ``sketch.props``]
- ``--method [ar|cegis|hybrid]``: the synthesis method  [default: ``ar``]

Options associated with the synthesis of finite-state controllers (FSCs) for a POMDP include:
- ``--fsc-memory-size INTEGER``    implicit memory size for (Dec-)POMDP FSCs [default: 1]
- ``--fsc-synthesis``: enables incremental synthesis of FSCs for a (Dec-)POMDP using iterative exploration of k-FSCs
- ``--posterior-aware``: enables the synthesis of posterior aware FSCs

SAYNT [6] and Storm associated options (pomdp-api branch of Storm and Stormpy are needed):
- ``--storm-pomdp``: enables the use of Storm features, this flag is necessary for the other options in this section to work
- ``--iterative-storm INTEGER INTEGER INTEGER``: runs the SAYNT algorithm, the parameters represent overall timeout, paynt timeout, storm timeout respectivelly. The recommended parameters for 15 minute runtime are 900 60 10
- ``--get-storm-result INTEGER``: runs PAYNT for specified amount of seconds and then runs Storm using the computed FSC at cut-offs
- ``--storm-options [cutoff|clip2|clip4|overapp|5mil|10mil|20mil|refine]``: sets the options for Storm [default: ``cutoff``]
- ``--prune-storm``: if enabled Storm results are used to prune the family of FSCs
- ``--unfold-strategy-storm [paynt|storm|cutoff]``: sets how the memory is unfolded [default: ``storm``]
- ``--use-storm-cutoffs``: if enabled the actions from cut-offs are considered in the prioritization and unfolding
- ``--export-synthesis PATH``: stores the synthesis result to speciefied PATH

Other options:
- ``--help``: shows the help message of the PAYNT and aborts
- ``--export [jani|drn|pomdp]``: exports the model to *.drn/*.pomdp and aborts


Here are various PAYNT calls:
```shell
python3 -m paynt models/archive/cav21-paynt/maze --props hard.props
python3 -m paynt models/archive/cav21-paynt/maze --props hard.props --method hybrid
python3 -m paynt models/archive/uai22-pomdp/grid-avoid-4-0
python3 -m paynt models/archive/uai22-pomdp/grid-avoid-4-0 --fsc-memory-size 2
python3 -m paynt models/archive/uai22-pomdp/grid-avoid-4-0 --fsc-memory-size 5
timeout 10s python3 -m paynt models/archive/uai22-pomdp/grid-avoid-4-0 --fsc-synthesis
python3 -m paynt models/archive/cav23-saynt/4x3-95 --fsc-synthesis --storm-pomdp --iterative-storm 180 60 10
python3 -m paynt models/archive/cav23-saynt/rocks-12 --fsc-synthesis --storm-pomdp --get-storm-result 0
```

The Python environment can be deactivated by running
```sh
deactivate
```

You might consider creating an alias (e.g. in your `.bashrc`) for simpler usage:
```shell
paynt() {
    source /path/to/your/venv/bin/activate
    python3 /path/to/your/paynt.py $@
    deactivate
}
```

# PAYNT tutorial

For instance, here is a simple PAYNT call:

```shell
python3 -m paynt models/archive/cav21-paynt/grid --props easy.props hybrid
```

Now we will investigate the __Grid__ model discussed in [1].
PAYNT inspects the content of this folder and locates the required files for the synthesis process: the sketch and the specification list.
In the example above, the sketch file is `models/archive/cav21-paynt/grid/sketch.templ` (in this case, `--sketch` option could have been omitted), the specification file is `models/archive/cav21-paynt/grid/easy.props` and the sketch does not have undefined constants, only holes.
Finally, the last argument specifies the selected synthesis method: `hybrid`.

## Getting started with PAYNT 

Having the tool installed, you can quickly test it by navigating to the tool folder, activating the Python environment and asking PAYNT to evaluate a simple synthesis problem:

```sh
cd /home/cav21/paynt
source env/bin/activate
python3 -m paynt models/archive/cav21-paynt/dpm-demo --method hybrid
```

The syntax of the command is described in more detail in the following chapters of this README.
For now, we can see that we ask PAYNT to look at the sketch (located in directory ``models/archive/cav21-paynt/dpm-demo``) for the dynamic power manager discussed in Section 2 in [1] and synthesize it wrt. specification in file ``models/archive/cav21-paynt/dpm-demo/sketch.props`` using the advanced hybrid approach.
The tool will print a series of log messages and, in the end, a short summary of the synthesis process, similar to the one below:

```
formula 1: R{"requests_lost"}<=1 [F "finished"]
optimal setting: formula: R{"power"}min=? [F "finished"]; direction: min; eps: 0.0

method: Hybrid, synthesis time: 12.39 s
number of holes: 7, family size: 12150
super MDP size: 1502, average MDP size: 1502, MPD checks: 2, iterations: 1
average DTMC size: 172, DTMC checks: 2708, iterations: 1354

optimal: 9100.064246
hole assignment: P1=1,P2=0,P3=0,P4=2,T1=0.0,T3=0.8,QMAX=5
```
The contents of such summary will be again discussed later.
Nonetheless, we can already notice the last line where tool reports a hole assignment that yields the optimal program.
The python environment can be deactivated by runnning

```sh
deactivate
```



## Synthesizing probabilistic programs with PAYNT



### Reading the output of PAYNT

Running PAYNT produces a sequence of log and a summary printed at the end of the synthesis process.
For instance, if we run

```sh
python3 -m paynt models/archive/cav21-paynt/dpm-demo --method hybrid
```
we obtain the following summary:

```shell
formula 1: R{"requests_lost"}<=1 [F "finished"]
optimal setting: formula: R{"power"}min=? [F "finished"]; direction: min; eps: 0.0

method: Hybrid, synthesis time: 67.62 s
number of holes: 7, family size: 12150
super MDP size: 1502, average MDP size: 956, MPD checks: 116, iterations: 59
average DTMC size: 234, DTMC checks: 14206, iterations: 7103

optimal: 9100.064246
hole assignment: P1=1,P2=2,P3=2,P4=2,T1=0.0,T3=0.8,QMAX=5
```

This summary contains information about the synthesized sketch as well as the results of the synthesis process.
The first lines repeat the synthesised specifications and, if included, the optimizing property.
Next, the synthesis was carried out using the hybrid method and it on our machine it took 68 seconds.
We can see that this particular DPM benchmark contains 7 holes (parameters) and 12K family members.
The following lines are statistics about deductive (MDP-based) or inductive (counterexample-based) analysis, including sizes of analyzed MDPs/DTMCs, number of particular model checking calls, overall iterations count etc.
Notice that only the hybrid method contains both MDP- and DTMC-related information since CEGIS never deals with MDPs, and AR works exclusively with MDPs.

Finally, the last lines show the synthesis result.
In our case, PAYNT printed a hole assignment yielding optimal solution as well as the induced optimal value.

### Sketching language

PAYNT takes as an input a sketch -- program description in `PRISM` language containing some undefined parameters (holes) with associated options from domains -- and a specification given as a list of temporal logic constraints (interpreted as a conjunction of these constrains) possibly including an optimal objective. Before explaining the sketching language, let us briefly present the key ideas of the `PRISM` language -- the full documentation of the language is available [in the PRISM manual](https://www.prismmodelchecker.org/manual/ThePRISMLanguage/Introduction).

A `PRISM` program consists of one or more reactive modules that may interact with each other using synchronisation. A module has a set of (bounded) variables that span its state space. Possible transitions between states of a module are described by a set of guarded commands of the form:

```
[action] guard -> prob_1 : update_1 + ... + prob_n : update_n; 
```

If the `guard` evaluates to true, an update of the variables is chosen according to the probability distribution given by expressions `p_1` through `p_n`. The `actions` are used to force two or more modules to make the command simultaneously (i.e. to synchronise).

Recall that the sketch is a `PRISM` program with holes and allows us to compactly describe a set of candidates program.
The holes can appear in guards and updates. Replacing each hole with one of its options yields a complete program with the semantics given by a finite-state Markov chain. 

We exemplify the usage of PAYNT by the following synthesis problem.

<img src="./doc/figures/dpm.jpg" alt="The server for request processing">

Consider a server for request processing depicted in Figure above.
Requests are generated (externally) in random intervals and upon arrival stored in a request queue of capacity Q<sub>max</sub>. 
When the queue is full, the request is lost.
The server has three profiles -- *sleeping*, *idle* and *active* -- that differ in their power consumption.
The requests are processed by the server only when it is in the active state.
Switching from a low-energy state into the active state requires additional energy as well as an additional random latency before the request can be processed. We further assume that the power consumption of request processing depends on the current queue size. The operation time of the server finite but given by a random process.

The goal of the synthesis process is to design power manager (PM) that controls the server. 
The PM observes the current queue size and then sets the desired power profile. 
We assume that the PM distinguishes between four queue occupancy levels determined by the threshold levels T<sub>1</sub>,T<sub>2</sub>, and T<sub>3</sub>. 
In other words, the PM observes the queue occupancy of the intervals: [0, T<sub>1</sub>], [T<sub>1</sub>, T<sub>2</sub>] etc. 
The values of these levels are unknown and thus are defined using four holes.
For each occupancy level, the PM changes to the associated power profile P<sub>1</sub>, ..., P<sub>4</sub> in {0,1,2}, where numbers 0 through 2 encode the profiles *sleeping*, *idle* and *active}*, respectively. 
The strategy which profile to used for the particular occupy is also unknown and thus defined using another four holes. 
Finally, the queue capacity Q<sub>max</sub> is also unknown and thus the sketch includes in total 8 holes.
In the sketch, the definition of the hole takes place outside of any module (typically in the beginning of the program) and must include its data type (int or double) as well as the domain:

```
// profiles desired at observation levels
// 0 - sleep, 1 - idle, 2 - active
hole int P1 in {0,1,2};
hole int P2 in {0,1,2};
hole int P3 in {0,1,2};
hole int P4 in {0,1,2};

// observation level thresholds
hole double T1 in {0.0,0.1,0.2,0.3,0.4};
hole double T2 in {0.5};
hole double T3 in {0.6,0.7,0.8};

// queue size
hole int QMAX in {1,2,3,4,5,6,7,8,9,10};
```

The following sketch fragment describes the module for the described power manager. The modules implementing other components of the server are omitted here for brevity -- the entire sketch is available in [this file](models/archive/cav21-paynt/dpm-demo/sketch.templ).

```
module PM
    pm  :  [0..2] init 0; // 0 - sleep, 1 - idle, 2 - active
    [tick0] q <= T1*QMAX -> (pm'=P1);
    [tick0] q > T1*QMAX & q <= T2*QMAX -> (pm'=P2);
    [tick0] q > T2*QMAX & q <= T3*QMAX -> (pm'=P3);
    [tick0] q > T3*QMAX -> (pm'=P4);
endmodule
```

Note that the domains of the holes defined above ensure that T<sub>1</sub> < T<sub>2</sub> < T<sub>3</sub>, however PAYNT further supports restrictions --- additional constraints on parameter combinations. 
The resulting sketch describes a *design space* of 10 x 5 x 4 x 3<sup>4</sup> = 16,200 different power managers where the average size of the underlying MC (of the complete system) is around 900 states. 

### Specification of the required behaviour

The goal is to find the concrete power manager, i.e., the instantiation of the holes, that minimizes power consumption while the expected number of lost requests during the operation time of the server is at most 1. 
In general, a specification is formalized as a list of temporal logic formulae in the [`PRISM` syntax](https://www.prismmodelchecker.org/manual/PropertySpecification/Introduction).
Here is a specification available within the benchmark directory [here](models/archive/cav21-paynt/dpm-demo/sketch.props):

```
R{"requests_lost"}<= 1 [ F "finished" ]
R{"power"}min=? [ F "finished" ]
```

We can see that the speicification file can additionally contain at most one optimizing property.
Furthermore, one can specify relative precision for satisfying such criterion (epsilon-optimality), e.g.

```
R{"power"}min{0.05}=? [ F "finished" ]
```

For the given sketch and specification, PAYNT effectively explores the design space and finds a hole assignment inducing a program that satisfies the specification, provided that such assignment exists.
Otherwise, it reports that such design does not exist.
If the specification also includes an optimizing criterion, PAYNT will find hole assignments that satisfies constraints in the specification __and__ has an optimal behaviour.

### Interpretation of the synthesis results

PAYNT produces the following output containing the hole assignment and the quality wrt. the specification of the corresponding program:

```
optimal: 9100.064246
hole assignment: P1=1,P2=2,P3=2,P4=2,T1=0.0,T3=0.8,QMAX=5
```

The obtained optimal power manager has queue capacity 5 with thresholds (after rounding) at 0, 2 and 4.
In addition, the power manager always maintains an active profile unless the request queue is empty, in which case the device is put into an idle state.
This solution guarantees expected number of lost requests to be at most one and has the power consumption of 9,100 units. 
To double-check that there are no controllers having expected power consumption less than 9100, we can modify the specification file `models/archive/cav21-paynt/dpm-demo/sketch.props` as follows:

```
R{"requests_lost"} <= 1 [ F "finished" ]
R{"power"}<=9100 [ F "finished" ]
```

Running PAYNT again (with hybrid synthesis approach) will produce the following result

```shell
formula 1: R{"requests_lost"}<=1 [F "finished"]
formula 2: R{"power"}<=9100 [F "finished"]

method: Hybrid, synthesis time: 67.52 s
number of holes: 7, family size: 12150
super MDP size: 1502, average MDP size: 962, MPD checks: 116, iterations: 59
average DTMC size: 237, DTMC checks: 14126, iterations: 7063

feasible: no
```
from which we can see that PAYNT indeed proved non-existence of a better solution.

We might further consider a more complex program sketch __Grid__ (discussed in [1]), where we synthesize controller for a robot in an unpredictable environment.

```shell
python3 -m paynt models/archive/cav21-paynt/grid --props easy.props --method hybrid
```

This sketch describes a family of 65K members, where each member has, on average 1225 states.
Even though this is a much larger family with much larger chains than in the sketch considered before, the pruning ability of the advanced hybrid approach allows PAYNT to handle this specification in a matter of seconds.
Meanwhile, one-by-one enumeration

```shell
python3 -m paynt models/archive/cav21-paynt/grid --props easy.props --method onebyone
```
might take up to 20 minutes.

## Testing PAYNT
The test dependencies are an optional extra, they are not installed with PAYNT itself.
From the root of the repository:

```shell
pip install ".[test]"
pytest tests                    # PAYNT tests
(cd payntbind && pytest tests)  # payntbind tests, run separately
```

Useful options:
- `-n auto` runs the tests in parallel.
- `--cov --cov-report=html` measures line and branch coverage of `paynt/`, including the runs of the command line in subprocesses, and writes the report to `htmlcov/`.
  Add `--cov-context=test` to see in the report which tests run each line.
- `-p randomly` runs the tests in random order, to find tests that depend on each other.

CI measures the coverage in the Release job on latest Storm, shows it in the job summary and uploads the HTML report as an artifact.

We also include mutation testing. It is slow, so run it on one package at a time:

```shell
pip install ".[mutation]"
mutmut run "paynt.parameter_space.*"
mutmut results               # the changes no test noticed
mutmut show <name>           # the diff of one of them
```

The first run takes about half an hour, since it runs all tests once to learn which tests reach which function; later runs reuse that and take minutes.
Not every surviving change is a missing test: some only change the speed of the search or the statistics that are printed.

We also have performance tests. `tests/performance/cases.toml` lists larger benchmarks (their models are in `models/performance-test/`), and `tests/performance/run.py` runs them and records the time, the peak memory, the number of iterations and the result of each.
To run some of them locally:

```shell
python tests/performance/run.py --only "ar-*"
```


---

# References

Most of the algorithms are described in:
- [3] Inductive Synthesis for Probabilistic Programs Reaches New Horizons by Roman Andriushchenko, Milan Ceska, Sebastian Junges, Joost-Pieter Katoen. In: TACAS'21.
- [4] Counterexample-Driven Synthesis for Probabilistic Program Sketches by Milan Ceska, Christian Hensel, Sebastian Junges, Joost-Pieter Katoen. In: FM'19.
- [5] Shepherding Hordes of Markov Chains by Milan Ceska, Nils Jansen, Sebastian Junges, Joost-Pieter Katoen. In: TACAS'19.
- [6] Inductive Synthesis of Finite-State Controllers for POMDPs by Roman Andriushchenko, Milan Ceska, Sebastian Junges, Joost-Pieter Katoen. In: UAI'22.
- [7] Search and Explore: Symbiotic Policy Synthesis in POMDPs by Roman Andriushchenko, Alexander Bork, Milan Ceska, Sebastian Junges, Joost-Pieter Katoen, Filip Macak. In: CAV'23.
- [8] Policies Grow on Trees: Model Checking Families of MDPs by Roman Andriushchenko, Milan Ceska, Sebastian Junges, and Filip Macak. In: ATVA'24.
- [9] Small Decision Trees for MDPs with Deductive Synthesis by by Roman Andriushchenko, Milan Ceska, Sebastian Junges, and Filip Macak. In: CAV'25.
