from __future__ import annotations

from typing import Any

import stormpy
import payntbind
import math
import operator

import logging

logger = logging.getLogger(__name__)


def construct_property(prop: Any, relative_error: float, use_exact: bool = False) -> Property:
    rf = prop.raw_formula
    player_index = None
    if rf.is_reward_operator and use_exact:
        raise ValueError("exact synthesis is not supported for reward properties")

    if not (rf.is_reward_operator or rf.is_probability_operator) and rf.is_game_formula:
        if use_exact:
            raise ValueError("exact synthesis is not supported for game properties")

        player_index = extract_player_index(rf)
        game_rf = rf
        rf = rf.subformula
        prop = stormpy.Property("", rf)
    assert rf.has_bound != rf.has_optimality_type, "optimizing formula contains a bound or a comparison formula does not"
    if rf.has_bound:
        prop = Property(prop, use_exact)
    else:
        prop = OptimalityProperty(prop, relative_error, use_exact)

    if player_index is not None:
        prop.game_optimizing_player = player_index
        prop.game_formula = game_rf
        alt_formula_str = f"<<{prop.game_optimizing_player}>> " + prop.formula_alt.__str__()
        formulas = stormpy.parse_properties(alt_formula_str)
        prop.game_formula_alt = formulas[0].raw_formula

    return prop


def extract_player_index(formula: Any) -> int:
    # TODO add support for multiple players in coalition
    string = formula.__str__()
    l_idx = string.index("<<")
    r_idx = string.index(">>")
    player_num = string[l_idx + len("<<") : r_idx]
    return int(player_num)


def construct_reward_property(reward_name: str, minimizing: bool, target_label: str) -> OptimalityProperty:
    direction = "min" if minimizing else "max"
    formula_str = 'R{"' + reward_name + '"}' + f'{direction}=? [F "{target_label}"]'
    formula = stormpy.parse_properties_without_context(formula_str)[0]
    return OptimalityProperty(formula, 0)


def construct_discounted_reward_property(reward_name: str, minimizing: bool, discount_factor: float) -> OptimalityProperty:
    direction = "min" if minimizing else "max"
    formula_str = 'R{"' + reward_name + '"}' + f"{direction}=? [Cdiscount={discount_factor}]"
    formula = stormpy.parse_properties_without_context(formula_str)[0]
    return OptimalityProperty(formula, 0)


def construct_specification(stormpy_properties: list[Any], relative_error: float = 0, use_exact: bool = False) -> Specification:
    """The canonical way to build a Specification from a list of raw stormpy properties."""
    Property.initialize(use_exact)
    properties = [construct_property(p, relative_error, use_exact) for p in stormpy_properties]
    return Specification(properties)


class Property:
    """Wrapper over a stormpy property."""

    # model checking environment (method & precision)
    environment: stormpy.Environment | None = None
    # model checking precision
    model_checking_precision: float = 1e-4
    # cap on minmax (value-iteration-family) solver iterations: without one, a non-converging direction
    # (observed on a reward property that never stabilizes) can hang indefinitely -- SMPMC in particular
    # calls model checking once per relevant theory-solver decision, far more often than AR/CEGIS call it
    # per node, so an unbounded solve is a much larger availability risk for it. 10000 matches molehill's
    # own default for the reference SMPMC implementation.
    max_minmax_iterations: int = 10000
    # --sound: apply no iteration caps. A capped solve that doesn't converge returns an unconverged value
    # (Storm only logs a warning), which can make pruning unsound; uncapped solves may instead not terminate.
    sound: bool = False

    @classmethod
    def set_model_checking_precision(cls, precision: float) -> None:
        cls.model_checking_precision = precision
        assert cls.environment is not None, "Property.initialize must be called before setting precision"
        payntbind.synthesis.set_precision_native(cls.environment.solver_environment.native_solver_environment, precision)
        payntbind.synthesis.set_precision_minmax(cls.environment.solver_environment.minmax_solver_environment, precision)

    @classmethod
    def initialize(cls, use_exact: bool = False) -> None:
        cls.environment = stormpy.Environment()
        cls.set_model_checking_precision(cls.model_checking_precision)
        if not cls.sound:
            payntbind.synthesis.set_max_iterations_minmax(cls.environment.solver_environment.minmax_solver_environment, cls.max_minmax_iterations)

        se = cls.environment.solver_environment
        # se.set_linear_equation_solver_type(stormpy.EquationSolverType.native)
        # se.set_linear_equation_solver_type(stormpy.EquationSolverType.gmmxx)
        # se.set_linear_equation_solver_type(stormpy.EquationSolverType.eigen)
        se.set_linear_equation_solver_type(stormpy.EquationSolverType.topological)

        if use_exact:
            se.minmax_solver_environment.method = stormpy.MinMaxMethod.policy_iteration
        else:
            se.minmax_solver_environment.method = stormpy.MinMaxMethod.optimistic_value_iteration

    @classmethod
    def model_check(cls, model: Any, formula: Any) -> Any:
        return stormpy.model_checking(model, formula, extract_scheduler=True, environment=cls.environment)

    @classmethod
    def compute_expected_visits(cls, model: Any) -> list[float]:
        result = stormpy.compute_expected_number_of_visits(cls.environment, model)
        return list(result.get_values())

    @staticmethod
    def above_model_checking_precision(a: Any, b: Any) -> bool:
        if isinstance(a, stormpy.Rational):
            return True
        return abs(a - b) > Property.model_checking_precision

    def __init__(self, prop: Any, use_exact: bool = False):
        self.property = prop
        rf = prop.raw_formula

        self.game_optimizing_player: int | None = None  # player index for game properties
        self.game_formula: Any = None
        # set alongside game_formula by construct_property, for game properties only -- declared here (not
        # just assigned dynamically there) so it has the same documented default as game_formula itself
        self.game_formula_alt: Any = None

        self.use_exact = use_exact
        # see discount_correction_factor; set only by paynt.parser.sketch's Cassandra branch
        self.discount_factor_correction: float | None = None

        # use comparison type to deduce optimizing direction
        comparison_type = rf.comparison_type
        self.minimizing = comparison_type in [stormpy.ComparisonType.LESS, stormpy.ComparisonType.LEQ]
        self.op = {
            stormpy.ComparisonType.LESS: operator.lt,
            stormpy.ComparisonType.LEQ: operator.le,
            stormpy.ComparisonType.GREATER: operator.gt,
            stormpy.ComparisonType.GEQ: operator.ge,
        }[comparison_type]

        # set threshold
        if use_exact:
            self.threshold = rf.threshold_expr.evaluate_as_rational()
            self.threshold_plus_precision = self.threshold
        else:
            self.threshold = rf.threshold_expr.evaluate_as_double()
            if self.minimizing:
                self.threshold_plus_precision = self.threshold + Property.model_checking_precision
            else:
                self.threshold_plus_precision = self.threshold - Property.model_checking_precision

        # construct quantitative formula (without bound) for explicit model checking
        # set optimality type
        self.formula = rf.clone()
        self.formula.remove_bound()
        if self.minimizing:
            self.formula.set_optimality_type(stormpy.OptimizationDirection.Minimize)
        else:
            self.formula.set_optimality_type(stormpy.OptimizationDirection.Maximize)
        self.formula_alt = Property.alt_formula(self.formula)

    @staticmethod
    def alt_formula(formula: Any) -> Any:
        """:return: formula with the opposite optimality type."""
        formula_alt = formula.clone()
        optimality_type = formula.optimality_type
        if optimality_type == stormpy.OptimizationDirection.Minimize:
            optimality_type = stormpy.OptimizationDirection.Maximize
        else:
            optimality_type = stormpy.OptimizationDirection.Minimize
        formula_alt.set_optimality_type(optimality_type)
        return formula_alt

    def __str__(self) -> str:
        return str(self.property.raw_formula)

    @property
    def reward(self) -> bool:
        return self.formula.is_reward_operator

    @property
    def is_discounted_reward(self) -> bool:
        return self.formula.is_reward_operator and self.formula.subformula.is_discounted_total_reward_formula

    def extract_discount_factor_from_formula(self) -> float:
        assert self.is_discounted_reward
        return self.formula.subformula.discount_factor_double

    @property
    def discount_correction_factor(self) -> float:
        """Cassandra-derived models add a synthetic, zero-reward initial state (see DecPomdp.cpp), so Cdiscount double-counts one discount step through it;
        discount_factor_correction (set only by paynt.parser.sketch's Cassandra branch) holds the value to divide back out.

        None means "no correction needed", not "unknown" -- a property can be discounted without this ever applying as this only applies for cases where the
        model comes from a Cassandra format.
        """
        if self.discount_factor_correction is None:
            return 1.0
        return 1.0 / self.discount_factor_correction

    @property
    def maximizing(self) -> bool:
        return not self.minimizing

    @property
    def is_until(self) -> bool:
        return self.formula.subformula.is_until_formula

    @property
    def has_game_formula(self) -> bool:
        return self.game_formula is not None

    def transform_until_to_eventually(self) -> None:
        if not self.is_until:
            return
        logger.info("converting until formula to eventually...")
        formula = payntbind.synthesis.transform_until_to_eventually(self.property.raw_formula)
        prop = stormpy.Property("", formula)
        self.__init__(prop, self.use_exact)  # type: ignore[misc]

    def property_copy(self) -> Any:
        return stormpy.Property("", self.property.raw_formula.clone())

    def copy(self) -> Property:
        new_property = Property(self.property_copy())
        new_property.discount_factor_correction = self.discount_factor_correction
        return new_property

    def result_valid(self, value: Any) -> bool:
        return not self.reward or value != math.inf

    def satisfies_threshold(self, value: Any) -> bool:
        return self.result_valid(value) and self.op(value, self.threshold)

    def satisfies_threshold_within_precision(self, value: Any) -> bool:
        return self.result_valid(value) and self.op(value, self.threshold_plus_precision)

    @property
    def can_be_improved(self) -> bool:
        return False

    def negate(self) -> Property:
        negated_formula = self.property.raw_formula.clone()
        negated_formula.comparison_type = {
            stormpy.ComparisonType.LESS: stormpy.ComparisonType.GEQ,
            stormpy.ComparisonType.LEQ: stormpy.ComparisonType.GREATER,
            stormpy.ComparisonType.GREATER: stormpy.ComparisonType.LEQ,
            stormpy.ComparisonType.GEQ: stormpy.ComparisonType.LESS,
        }[negated_formula.comparison_type]
        stormpy_property_negated = stormpy.Property("", negated_formula)
        negated_property = Property(stormpy_property_negated)
        negated_property.discount_factor_correction = self.discount_factor_correction
        return negated_property

    def get_target_label(self) -> str:
        target = self.formula.subformula.subformula
        if isinstance(target, stormpy.logic.AtomicLabelFormula):
            target_label = target.label
        elif isinstance(target, stormpy.logic.AtomicExpressionFormula):
            target_label = str(target)
        else:
            raise ValueError(f"unknown type of target expression {str(target)}, expected atomic label or atomic expression")
        return target_label

    def get_reward_name(self) -> str:
        assert self.reward
        return self.formula.reward_name

    def transform_to_optimality_formula(self, prism: Any) -> Any:
        direction = "min" if self.minimizing else "max"
        if self.reward:
            if isinstance(self.formula.subformula.subformula, stormpy.logic.AtomicLabelFormula):
                formula_str = f'R{{"{self.get_reward_name()}"}}{direction}=? [F "{self.get_target_label()}"]'
            else:
                formula_str = f'R{{"{self.get_reward_name()}"}}{direction}=? [F {self.get_target_label()}]'
        else:
            if isinstance(self.formula.subformula.subformula, stormpy.logic.AtomicLabelFormula):
                formula_str = f'P{direction}=? [F "{self.get_target_label()}"]'
            else:
                formula_str = f"P{direction}=? [F {self.get_target_label()}]"
        return stormpy.parse_properties_for_prism_program(formula_str, prism)[0]


class OptimalityProperty(Property):
    """Optimality property can remember current optimal value and adapt the corresponding threshold wrt epsilon."""

    def __init__(self, prop: Any, epsilon: float = 0, use_exact: bool = False):
        self.property = prop
        rf = prop.raw_formula

        self.game_optimizing_player: int | None = None  # player index for game properties
        self.game_formula: Any = None
        # set alongside game_formula by construct_property, for game properties only -- declared here (not
        # just assigned dynamically there) so it has the same documented default as game_formula itself
        self.game_formula_alt: Any = None

        self.use_exact = use_exact
        # see Property.discount_correction_factor; set only by paynt.parser.sketch's Cassandra branch
        self.discount_factor_correction: float | None = None

        # use comparison type to deduce optimizing direction
        if rf.optimality_type == stormpy.OptimizationDirection.Minimize:
            self.minimizing = True
            self.op = operator.lt
        else:
            self.minimizing = False
            self.op = operator.gt

        # construct quantitative formula (without bound) for explicit model checking
        self.formula = rf.clone()
        self.formula_alt = Property.alt_formula(self.formula)

        # additional optimality stuff
        self.optimum = None
        if use_exact:
            self.epsilon = stormpy.Rational(epsilon)
        else:
            self.epsilon = epsilon

        self.reset()

    def __str__(self) -> str:
        eps = f"[eps = {self.epsilon}]" if self.epsilon > 0 else ""
        return f"{str(self.formula)} {eps}"

    def copy(self) -> OptimalityProperty:
        new_property = OptimalityProperty(self.property_copy(), self.epsilon, self.use_exact)
        new_property.discount_factor_correction = self.discount_factor_correction
        return new_property

    def reset(self) -> None:
        self.optimum = None
        if self.minimizing:
            if self.use_exact:
                self.threshold = stormpy.Rational(2)  # TODO: does not work for rewards
            else:
                self.threshold = math.inf
        else:
            if self.use_exact:
                self.threshold = stormpy.Rational(-1)  # TODO: does not work for rewards
            else:
                self.threshold = -math.inf

    def meets_op(self, a: Any, b: Any) -> bool:
        """For optimality objective, we want to accept improvements above model checking precision."""
        return b is None or (Property.above_model_checking_precision(a, b) and self.op(a, b))

    def satisfies_threshold(self, value: Any) -> bool:
        return self.result_valid(value) and self.meets_op(value, self.threshold)

    def improves_optimum(self, value: Any) -> bool:
        return self.result_valid(value) and self.meets_op(value, self.optimum)

    def update_optimum(self, optimum: Any) -> None:
        self.optimum = optimum
        if self.minimizing:
            self.threshold = optimum * (1 - self.epsilon)
        else:
            self.threshold = optimum * (1 + self.epsilon)

    def suboptimal_value(self) -> Any:
        assert self.optimum is not None
        if self.minimizing:
            return self.optimum * (1 + self.model_checking_precision)
        return self.optimum * (1 - self.model_checking_precision)

    def transform_until_to_eventually(self) -> None:
        if not self.is_until:
            return
        logger.info("converting until formula to eventually...")
        formula = payntbind.synthesis.transform_until_to_eventually(self.property.raw_formula)
        prop = stormpy.Property("", formula)
        self.__init__(prop, self.epsilon, self.use_exact)  # type: ignore[misc]

    @property
    def can_be_improved(self) -> bool:
        return not (not self.reward and self.minimizing and self.threshold == 0)

    def negate(self) -> OptimalityProperty:
        negated_formula = self.property.raw_formula.clone()
        negate_optimality_type = {
            stormpy.OptimizationDirection.Minimize: stormpy.OptimizationDirection.Maximize,
            stormpy.OptimizationDirection.Maximize: stormpy.OptimizationDirection.Minimize,
        }[negated_formula.optimality_type]
        negated_formula.set_optimality_type(negate_optimality_type)
        stormpy_property_negated = stormpy.Property("", negated_formula)
        negated_property = OptimalityProperty(stormpy_property_negated, self.epsilon)
        negated_property.discount_factor_correction = self.discount_factor_correction
        return negated_property


class Specification:
    def __init__(self, properties: list[Property]):
        self.constraints: list[Property] = []
        self.optimality: OptimalityProperty | None = None

        # sort the properties
        optimalities = []
        for p in properties:
            if type(p) is Property:
                self.constraints.append(p)
            if type(p) is OptimalityProperty:
                optimalities.append(p)
        assert len(optimalities) <= 1, "multiple optimality objectives were specified"
        if optimalities:
            self.optimality = optimalities[0]

    def __str__(self) -> str:
        s = ""
        if self.constraints:
            s += "constraints: " + ",".join([str(c) for c in self.constraints]) + "; "
        if self.optimality is not None:
            s += "optimality: " + str(self.optimality)
        return s

    def copy(self) -> Specification:
        properties = [p.copy() for p in self.all_properties()]
        return Specification(properties)

    def reset(self) -> None:
        if self.optimality is not None:
            self.optimality.reset()

    @property
    def has_optimality(self) -> bool:
        return self.optimality is not None

    @property
    def num_properties(self) -> int:
        return len(self.constraints) + (1 if self.has_optimality else 0)

    @property
    def is_single_property(self) -> bool:
        return self.num_properties == 1

    def all_properties(self) -> list[Property]:
        properties = list(self.constraints)
        if self.optimality is not None:
            properties += [self.optimality]
        return properties

    def all_constraint_indices(self) -> range:
        return range(len(self.constraints))

    def stormpy_properties(self) -> list[Any]:
        return [p.property for p in self.all_properties()]

    def stormpy_formulae(self) -> list[Any]:
        return [p.formula for p in self.all_properties()]

    def contains_until_properties(self) -> bool:
        return any(p.is_until for p in self.all_properties())

    def transform_until_to_eventually(self) -> None:
        for p in self.all_properties():
            p.transform_until_to_eventually()

    def check(self) -> None:
        # TODO
        pass

    def can_be_improved(self) -> bool:
        return any(prop.can_be_improved for prop in self.all_properties())

    @property
    def contains_maximizing_reward_properties(self) -> bool:
        return any(c.reward and not c.minimizing for c in self.all_properties())

    def negate(self) -> Specification:
        properties_negated = [p.negate() for p in self.all_properties()]
        return Specification(properties_negated)

    def rewrap(self, new_properties: list[Any], use_exact: bool = False) -> Specification:
        """Rebuild a Specification from new raw stormpy properties (e.g. after PRISM->JANI translation changed their atoms), preserving each property's paynt-
        level type (Property vs OptimalityProperty) and epsilon.

        :param new_properties: raw stormpy properties, same order and length as self.all_properties()
        """
        old_properties = self.all_properties()
        assert len(new_properties) == len(old_properties)
        properties_rewrapped = []
        for prop_old, prop_new in zip(old_properties, new_properties, strict=False):
            if type(prop_old) is Property:
                p = Property(prop_new, use_exact)
            else:
                assert isinstance(prop_old, OptimalityProperty)
                p = OptimalityProperty(prop_new, prop_old.epsilon, use_exact)
            properties_rewrapped.append(p)
        return Specification(properties_rewrapped)
