"""The conflicts of CEGIS's counterexample generators."""

import pytest
import stormpy

import paynt.parser.sketch
import paynt.specification.property
import paynt.synthesizer.conflict_generator.auto
import paynt.synthesizer.search_node

# each hole costs 1 when it is 1: the design H0=1, H1=1 costs 2
TWO_COSTS = """{model_type}

hole int H0 in {{0,1}};
hole int H1 in {{0,1}};

module m
    s : [0..2] init 0;
    [free0] s=0 & H0=0 -> (s'=1);
    [paid0] s=0 & H0=1 -> (s'=1);
    [free1] s=1 & H1=0 -> (s'=2);
    [paid1] s=1 & H1=1 -> (s'=2);
    [] s=2 -> true;
    {mdp_only}[idle] s=2 -> true;
endmodule

label "done" = s=2;

rewards "cost"
    [paid0] true : 1;
    [paid1] true : 1;
endrewards
"""

# H0=1 enters a loop that ends in "bad" half of the time, H1 comes after it
LEAKY_LOOP = """{model_type}

hole int H0 in {{0,1}};
hole int H1 in {{0,1}};

module m
    s : [0..5] init 0;
    [safe] s=0 & H0=0 -> (s'=3);
    [loop] s=0 & H0=1 -> (s'=1);
    [] s=1 -> (s'=2);
    {mdp_only}[idle] s=1 -> (s'=2);
    [] s=2 -> 0.998 : (s'=1) + 0.001 : (s'=3) + 0.001 : (s'=5);
    [stay1] s=3 & H1=0 -> (s'=4);
    [fail1] s=3 & H1=1 -> (s'=5);
    [] s>=4 -> true;
endmodule

label "bad" = s=5;
"""

# in the MDP, the idle choice, which no hole decides, leaves every design an MDP
MODEL_TYPES = ["dtmc", "mdp"]


def load(tmp_path, template, props, model_type):
    (tmp_path / "sketch.templ").write_text(template.format(model_type=model_type, mdp_only="" if model_type == "mdp" else "// "))
    (tmp_path / "sketch.props").write_text(props)
    return paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))


def conflict_of_the_design_h0_1_h1_1(factory, task, model_type):
    colored_mdp = factory.build()
    space = colored_mdp.parameter_space
    node = paynt.synthesizer.search_node.SearchNode(space)
    node.mdp, node.selected_choices = colored_mdp.build(space)
    generator = paynt.synthesizer.conflict_generator.auto.ConflictGeneratorAuto(colored_mdp, task)
    generator.initialize()
    assignment = space.assume_options_copy([[1], [1]])
    design = colored_mdp.build_assignment(assignment)
    assert stormpy.models.SparseDtmc.is_instantiation(design.model) == (model_type == "dtmc")
    request = (0, task.specification.constraints[0], None)
    [conflict] = generator.construct_conflicts(node, assignment, design, [request])
    return sorted(space.parameter_name(parameter) for parameter in conflict)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_a_cost_equal_to_a_strict_bound_is_a_conflict(tmp_path, model_type):
    # H0=1 alone costs 1, which already violates R<1
    factory, task = load(tmp_path, TWO_COSTS, 'R{"cost"}<1 [F "done"]', model_type)
    assert conflict_of_the_design_h0_1_h1_1(factory, task, model_type) == ["H0"]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_a_cost_equal_to_a_non_strict_bound_is_no_conflict(tmp_path, model_type):
    # H0=1, H1=0 costs 1 and satisfies R<=1
    factory, task = load(tmp_path, TWO_COSTS, 'R{"cost"}<=1 [F "done"]', model_type)
    assert conflict_of_the_design_h0_1_h1_1(factory, task, model_type) == ["H0", "H1"]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_the_loop_alone_is_a_conflict(tmp_path, model_type):
    factory, task = load(tmp_path, LEAKY_LOOP, 'P<=0.4 [F "bad"]', model_type)
    assert conflict_of_the_design_h0_1_h1_1(factory, task, model_type) == ["H0"]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_the_generators_check_with_paynt_s_solver_environment(tmp_path, model_type, monkeypatch):
    """Stopped after one iteration, the loop's probability of "bad" stays far below 0.4, so the loop alone is no conflict."""
    factory, task = load(tmp_path, LEAKY_LOOP, 'P<=0.4 [F "bad"]', model_type)
    environment = stormpy.Environment()
    environment.solver().set_linear_equation_solver_type(stormpy.environments.EquationSolverType.NATIVE)
    environment.solver().native().method = stormpy.environments.NativeLinearEquationSolverMethod.POWER
    environment.solver().native().maximal_number_of_iterations = 1
    environment.solver().min_max().method = stormpy.environments.MinMaxMethod.VALUE_ITERATION
    environment.solver().min_max().maximal_number_of_iterations = 1
    monkeypatch.setattr(paynt.specification.property.Property, "environment", environment)
    assert conflict_of_the_design_h0_1_h1_1(factory, task, model_type) == ["H0", "H1"]
