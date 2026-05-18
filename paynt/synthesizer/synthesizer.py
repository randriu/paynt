import paynt.synthesizer.statistic
import paynt.utils.timer

import logging
logger = logging.getLogger(__name__)


class FamilyEvaluation:
    '''Result associated with a family after its evaluation. '''
    def __init__(self, family, value, sat, policy):
        self.family = family
        self.value = value
        self.sat = sat
        self.policy = policy


class Synthesizer:

    # base filename (i.e. without extension) to export synthesis result
    export_synthesis_filename_base = None

    @staticmethod
    def choose_synthesizer(quotient, method, fsc_synthesis=False, storm_control=None):

        # hiding imports here to avoid mutual top-level imports
        import paynt.quotient.pomdp
        import paynt.quotient.decpomdp
        import paynt.quotient.mdp_family
        import paynt.quotient.posmg
        import paynt.synthesizer.synthesizer_onebyone
        import paynt.synthesizer.synthesizer_ar
        import paynt.synthesizer.synthesizer_cegis
        import paynt.synthesizer.synthesizer_hybrid
        import paynt.synthesizer.synthesizer_multicore_ar
        import paynt.synthesizer.synthesizer_pomdp
        import paynt.synthesizer.synthesizer_decpomdp
        import paynt.synthesizer.synthesizer_posmg
        import paynt.synthesizer.policy_tree

        from paynt.dt import DtColoredMdpFactory, DtSynthesizer
        from paynt.dt.dtnest import DtNest

        if isinstance(quotient, paynt.quotient.pomdp_family.PomdpFamilyQuotient):
            logger.info("nothing to do with the POMDP sketch, aborting...")
            exit(0)
        if isinstance(quotient, DtColoredMdpFactory):
            return DtSynthesizer(quotient)
        # FSC synthesis for POMDPs
        if isinstance(quotient, paynt.quotient.pomdp.PomdpQuotient) and fsc_synthesis:
            return paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(quotient, method, storm_control)
        # FSC synthesis for Dec-POMDPs
        if isinstance(quotient, paynt.quotient.decpomdp.DecPomdpQuotient) and fsc_synthesis:
            return paynt.synthesizer.synthesizer_decpomdp.SynthesizerDecPomdp(quotient)
        # Policy Tree synthesis for family of MDPs
        if isinstance(quotient, paynt.quotient.mdp_family.MdpFamilyQuotient):
            if method == "onebyone":
                return paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(quotient)
            else:
                return paynt.synthesizer.policy_tree.SynthesizerPolicyTree(quotient)
        # FSC synthesis for POSMGs
        if isinstance(quotient, paynt.quotient.posmg.PosmgQuotient) and fsc_synthesis:
            return paynt.synthesizer.synthesizer_posmg.SynthesizerPosmg(quotient)

        # synthesis engines
        if method == "onebyone":
            return paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(quotient)
        if method == "ar":
            return paynt.synthesizer.synthesizer_ar.SynthesizerAR(quotient)
        if method == "cegis":
            return paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS(quotient)
        if method == "hybrid":
            return paynt.synthesizer.synthesizer_hybrid.SynthesizerHybrid(quotient)
        if method == "ar_multicore":
            return paynt.synthesizer.synthesizer_multicore_ar.SynthesizerMultiCoreAR(quotient)
        raise ValueError("invalid method name")


    def __init__(self, quotient):
        self.quotient = quotient
        self.stat = None
        self.synthesis_timer = None
        self.explored = None
        self.best_assignment = None
        self.best_assignment_value = None

    @property
    def method_name(self):
        ''' to be overridden '''
        pass

    def time_limit_reached(self):
        if (self.synthesis_timer is not None and self.synthesis_timer.time_limit_reached()) or \
            paynt.utils.timer.GlobalTimer.time_limit_reached():
            logger.info("time limit reached, aborting...")
            return True
        return False

    def memory_limit_reached(self):
        if paynt.utils.timer.GlobalMemoryLimit.limit_reached():
            logger.info("memory limit reached, aborting...")
            return True
        return False

    def resource_limit_reached(self):
        return self.time_limit_reached() or self.memory_limit_reached()

    def set_optimality_threshold(self, optimum_threshold):
        if self.quotient.specification.has_optimality and optimum_threshold is not None:
            self.quotient.specification.optimality.update_optimum(optimum_threshold)
            logger.debug(f"optimality threshold set to {optimum_threshold}")

    def explore(self, family):
        self.explored += family.size

    def evaluate_all(self, family, prop, keep_value_only=False):
        ''' to be overridden '''
        pass

    def export_evaluation_result(self, evaluations, export_filename_base):
        ''' to be overridden '''
        pass

    def evaluate(self, family=None, prop=None, keep_value_only=False, print_stats=True):
        '''
        Evaluate each member of the family wrt the given property.
        :param family if None, then the design space of the quotient will be used
        :param prop if None, then the default property of the quotient will be used
            (assuming single-property specification)
        :param keep_value_only if True, only value will be associated with the family
        :param print_stats if True, synthesis statistic will be printed
        :param export_filename_base base filename used to export the evaluation results
        :returns a list of (family,evaluation) pairs
        '''
        if family is None:
            family = self.quotient.family
        if prop is None:
            prop = self.quotient.get_property()

        self.stat = paynt.synthesizer.statistic.Statistic(self)
        self.explored = 0
        logger.info("evaluation initiated, design space: {}".format(family.size))
        self.stat.start(family)
        evaluations = self.evaluate_all(family, prop, keep_value_only)
        self.stat.finished_evaluation(evaluations)
        logger.info("evaluation finished")

        if self.export_synthesis_filename_base is not None:
            self.export_evaluation_result(evaluations, self.export_synthesis_filename_base)

        if print_stats:
            self.stat.print()

        return evaluations


    def synthesize_one(self, family):
        ''' to be overridden '''
        pass

    def synthesize(
        self, family=None, optimum_threshold=None, keep_optimum=False, return_all=False, print_stats=True, timeout=None
    ):
        '''
        :param family family of assignment to search in
        :param families alternatively, a list of families can be given
        :param optimum_threshold known bound on the optimum value
        :param keep_optimum if True, the optimality specification will not be reset upon finish
        :param return_all if True and the synthesis returns a family, all assignments will be returned instead of an
            arbitrary one
        :param print_stats if True, synthesis stats will be printed upon completion
        :param timeout synthesis time limit, seconds
        '''
        if family is None:
            family = self.quotient.family
        if family.constraint_indices is None:
            family.constraint_indices = list(range(len(self.quotient.specification.constraints)))

        self.set_optimality_threshold(optimum_threshold)
        self.synthesis_timer = paynt.utils.timer.Timer(timeout)
        self.synthesis_timer.start()
        self.stat = paynt.synthesizer.statistic.Statistic(self)
        self.explored = 0
        self.stat.start(family)
        self.synthesize_one(family)
        if self.best_assignment is not None and self.best_assignment.size > 1 and not return_all:
            self.best_assignment = self.best_assignment.pick_any()
        self.stat.finished_synthesis()
        if self.best_assignment is not None:
            logger.info("printing synthesized assignment below:")
            logger.info(self.best_assignment)

        if self.best_assignment is not None and self.best_assignment.size == 1:
            dtmc = self.quotient.build_assignment(self.best_assignment)
            result = dtmc.check_specification(self.quotient.specification)
            logger.info(f"double-checking specification satisfiability: {result}")

        if print_stats:
            self.stat.print()

        assignment = self.best_assignment
        if not keep_optimum:
            self.best_assignment = None
            self.best_assignment_value = None
            self.quotient.specification.reset()

        return assignment


    def run(self, optimum_threshold=None):
        return self.synthesize(optimum_threshold=optimum_threshold)
