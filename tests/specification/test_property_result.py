import paynt.specification.property_result


class _FakeConstraintResult:
    def __init__(self, sat, primary_selection=None):
        self.sat = sat
        self.primary_selection = primary_selection


class _FakeOptimalityResult:
    def __init__(self, improves_optimum=None, value=None, can_improve=None, primary_selection=None):
        self.improves_optimum = improves_optimum
        self.value = value
        # improves_optimum and value are read by accepting_dtmc, can_improve by undecided_results
        self.can_improve = can_improve
        self.primary_selection = primary_selection


def _spec_result(constraint_sats, optimality_result=None):
    """An MdpSpecificationResult with the given constraint verdicts (None is undecided); it inherits accepting_dtmc."""
    spec_result = paynt.specification.property_result.MdpSpecificationResult()
    spec_result.constraints_result = paynt.specification.property_result.ConstraintsResult([_FakeConstraintResult(sat) for sat in constraint_sats])
    spec_result.optimality_result = optimality_result
    return spec_result


class TestAcceptingDtmc:
    """accepting_dtmc returns the pair (accepting, improving value), so a caller has to unpack it: a tuple is always true."""

    def test_unsat_constraints_return_false_and_must_not_be_used_as_a_bare_condition(self):
        result = _spec_result(constraint_sats=[False])
        assert result.accepting_dtmc(specification=None) == (False, None)

    def test_sat_constraints_no_optimality_returns_true_and_none(self):
        result = _spec_result(constraint_sats=[True])
        assert result.accepting_dtmc(specification=None) == (True, None)

    def test_sat_constraints_optimality_not_improving_returns_false(self):
        """An assignment that satisfies the constraints but does not improve the optimum found so far is not accepted."""
        result = _spec_result(constraint_sats=[True], optimality_result=_FakeOptimalityResult(improves_optimum=False))
        accepting, improving_value = result.accepting_dtmc(specification=None)
        assert accepting is False
        assert improving_value is None

    def test_sat_constraints_optimality_improving_returns_true_and_value(self):
        result = _spec_result(constraint_sats=[True], optimality_result=_FakeOptimalityResult(improves_optimum=True, value=0.75))
        assert result.accepting_dtmc(specification=None) == (True, 0.75)


class TestUndecidedResults:
    """undecided_results lists every property still undecided: the undecided constraints in order, then the objective if it can still improve.

    Splitting
    with several constraints compares all of them (https://www.jair.org/index.php/jair/article/view/16593, Section 3.3).
    """

    def test_single_undecided_constraint_no_optimality_matches_undecided_result(self):
        spec_result = _spec_result(constraint_sats=[True, None])
        undecided_constraint = spec_result.constraints_result.results[1]
        assert spec_result.undecided_results() == [undecided_constraint]
        # with one undecided property it is the one undecided_result gives
        assert spec_result.undecided_results() == [spec_result.undecided_result()]

    def test_multiple_undecided_constraints_returned_in_ascending_index_order(self):
        spec_result = _spec_result(constraint_sats=[None, True, None])
        c0 = spec_result.constraints_result.results[0]
        c2 = spec_result.constraints_result.results[2]
        assert spec_result.undecided_results() == [c0, c2]

    def test_optimality_appended_last_when_it_can_still_improve(self):
        opt = _FakeOptimalityResult(can_improve=True)
        spec_result = _spec_result(constraint_sats=[None], optimality_result=opt)
        c0 = spec_result.constraints_result.results[0]
        assert spec_result.undecided_results() == [c0, opt]

    def test_optimality_omitted_when_it_cannot_improve(self):
        opt = _FakeOptimalityResult(can_improve=False)
        spec_result = _spec_result(constraint_sats=[None], optimality_result=opt)
        c0 = spec_result.constraints_result.results[0]
        assert spec_result.undecided_results() == [c0]

    def test_no_undecided_constraints_optimality_only_matches_undecided_result(self):
        opt = _FakeOptimalityResult(can_improve=True)
        spec_result = _spec_result(constraint_sats=[True], optimality_result=opt)
        assert spec_result.undecided_results() == [opt]
        assert spec_result.undecided_results() == [spec_result.undecided_result()]
