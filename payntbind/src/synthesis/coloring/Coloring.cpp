#include "Coloring.h"

#include <algorithm>

namespace synthesis {

Coloring::Coloring(
    Family const& family, std::vector<uint64_t> const& row_groups,
    std::vector<std::vector<std::pair<uint64_t,uint64_t>>> choice_to_assignment
) : family(family), choice_to_assignment(choice_to_assignment), row_groups(row_groups) {
    
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


    // sorted lists rather than bit vectors over all holes, which are quadratic when every state has its own hole (robust search over an MDP family)
    auto num_states = row_groups.size()-1;
    state_to_holes.resize(num_states);
    for(uint64_t state = 0; state<num_states; ++state) {
        auto& holes = state_to_holes[state];
        for(uint64_t choice = row_groups[state]; choice<row_groups[state+1]; ++choice) {
            for(auto const& [hole,option]: choice_to_assignment[choice]) {
                holes.push_back(hole);
            }
        }
        std::sort(holes.begin(), holes.end());
        holes.erase(std::unique(holes.begin(), holes.end()), holes.end());
    }
}

const uint64_t Coloring::numChoices() const {
    return choice_to_assignment.size();
}

std::vector<std::vector<std::pair<uint64_t,uint64_t>>> const& Coloring::getChoiceToAssignment() const {
    return choice_to_assignment;
}

std::vector<std::vector<uint64_t>> const& Coloring::getStateToHoles() const {
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


std::vector<uint64_t> Coloring::statesWithoutChoice(Family const& subfamily, std::vector<uint64_t> const& states) const {
    std::vector<std::vector<uint64_t>> groups;
    for(auto state: states) {
        std::vector<uint64_t> choices;
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            choices.push_back(choice);
        }
        groups.push_back(choices);
    }
    std::vector<uint64_t> without_choice;
    for(auto group: groupsWithoutChoice(subfamily, groups)) {
        without_choice.push_back(states[group]);
    }
    return without_choice;
}

std::vector<uint64_t> Coloring::groupsWithoutChoice(Family const& subfamily, std::vector<std::vector<uint64_t>> const& groups) const {
    std::vector<BitVector> options;
    for(uint64_t hole = 0; hole < subfamily.numHoles(); ++hole) {
        options.push_back(subfamily.holeOptionsMask(hole));
    }
    std::vector<uint64_t> without_choice;
    for(uint64_t group = 0; group < groups.size(); ++group) {
        if(not someColorHolds(groups[group], options)) {
            without_choice.push_back(group);
        }
    }
    return without_choice;
}

bool Coloring::someColorHolds(std::vector<uint64_t> const& choices, std::vector<BitVector>& options) const {
    std::vector<uint64_t> possible;
    for(auto choice: choices) {
        bool enabled_by_some = true;
        bool enabled_by_all = true;
        for(auto const& [hole,option]: choice_to_assignment[choice]) {
            if(not options[hole].get(option)) {
                enabled_by_some = false;
                break;
            }
            if(options[hole].getNumberOfSetBits() > 1) {
                enabled_by_all = false;
            }
        }
        if(not enabled_by_some) {
            continue;
        }
        if(enabled_by_all) {
            return true;
        }
        possible.push_back(choice);
    }
    if(possible.empty()) {
        return false;
    }
    // no color holds for every assignment: try each option of a hole of theirs that has several
    uint64_t split_hole = 0;
    bool found = false;
    for(auto choice: possible) {
        for(auto const& [hole,option]: choice_to_assignment[choice]) {
            if(options[hole].getNumberOfSetBits() > 1) {
                split_hole = hole;
                found = true;
                break;
            }
        }
        if(found) {
            break;
        }
    }
    BitVector saved = options[split_hole];
    bool holds = true;
    for(auto option: saved) {
        options[split_hole] = BitVector(saved.size(), false);
        options[split_hole].set(option, true);
        if(not someColorHolds(possible, options)) {
            holds = false;
            break;
        }
    }
    options[split_hole] = saved;
    return holds;
}

}
