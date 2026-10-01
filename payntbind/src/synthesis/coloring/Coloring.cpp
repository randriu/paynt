#include "Coloring.h"

namespace synthesis {

Coloring::Coloring(
    Family const& family, std::vector<uint64_t> const& row_groups,
    std::vector<std::vector<std::pair<uint64_t,uint64_t>>> choice_to_assignment
) : family(family), choice_to_assignment(choice_to_assignment) {
    
    auto num_choices = numChoices();
    colored_choices.resize(num_choices,false);
    uncolored_choices.resize(num_choices,false);
    for(uint64_t choice = 0; choice<num_choices; ++choice) {
        if(choice_to_assignment[choice].empty()) {
            uncolored_choices.set(choice,true);
        } else {
            colored_choices.set(choice,true);
        }
    }


    auto num_holes = family.numHoles();
    choice_to_holes.resize(num_choices);
    for(uint64_t choice = 0; choice<num_choices; ++choice) {
        choice_to_holes[choice] = BitVector(num_holes,false);
        for(auto const& [hole,option]: choice_to_assignment[choice]) {
            choice_to_holes[choice].set(hole,true);
        }
    }


    auto num_states = row_groups.size()-1;
    state_to_holes.resize(num_states);
    for(uint64_t state = 0; state<num_states; ++state) {
        state_to_holes[state] = BitVector(num_holes,false);
        for(uint64_t choice = row_groups[state]; choice<row_groups[state+1]; ++choice) {
            state_to_holes[state] = state_to_holes[state] | choice_to_holes[choice]; 
        }
    }
}

const uint64_t Coloring::numChoices() const {
    return choice_to_assignment.size();
}

std::vector<std::vector<std::pair<uint64_t,uint64_t>>> const& Coloring::getChoiceToAssignment() const {
    return choice_to_assignment;
}

std::vector<BitVector> const& Coloring::getStateToHoles() const {
    return state_to_holes;
}

BitVector Coloring::selectCompatibleChoices(Family const& subfamily) const {
    return selectCompatibleChoices(subfamily, colored_choices);
}

BitVector Coloring::selectCompatibleChoices(Family const& subfamily, BitVector const& base_choices) const {
    auto selection = BitVector(uncolored_choices);
    auto candidate_choices = colored_choices & base_choices;
    for(auto choice: candidate_choices) {
        if(subfamily.includesAssignment(choice_to_assignment[choice])) {
            selection.set(choice,true);
        }
    }
    return selection;
}



std::vector<BitVector> Coloring::collectHoleOptionsMask(BitVector const& choices) const {

    std::vector<BitVector> hole_options_mask(family.numHoles());
    for(uint64_t hole = 0; hole < family.numHoles(); ++hole) {
        hole_options_mask[hole] = BitVector(family.holeNumOptionsTotal(hole),false);
    }
    for(auto choice: choices) {
        for(auto const& [hole,option]: choice_to_assignment[choice]) {
            hole_options_mask[hole].set(option,true);
        }
    }
    return hole_options_mask;
}


std::vector<std::vector<uint64_t>> Coloring::collectHoleOptions(BitVector const& choices) const {
    auto hole_options_mask = collectHoleOptionsMask(choices);
    std::vector<std::vector<uint64_t>> hole_options(family.numHoles());
    for(uint64_t hole = 0; hole < family.numHoles(); ++hole) {
        for(auto option: hole_options_mask[hole]) {
            hole_options[hole].push_back(option);
        }
    }
    return hole_options;
}

}