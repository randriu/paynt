#include "Coloring.h"

#include <iostream>


namespace synthesis {

Family::Family(Family const& other) {
    hole_options = std::vector<std::vector<uint64_t>>(other.numHoles());
    hole_options_mask = std::vector<BitVector>(other.numHoles());
    for(uint64_t hole = 0; hole < numHoles(); ++hole) {
        hole_options[hole] = other.holeOptions(hole);
        hole_options_mask[hole] = other.holeOptionsMask(hole);
    }
}

uint64_t Family::numHoles() const {
    return hole_options.size();
}

uint64_t Family::addHole(uint64_t num_options) {
    uint64_t hole = numHoles();
    hole_options_mask.push_back(BitVector(num_options,true));
    std::vector<uint64_t> options(num_options);
    for(uint64_t option=0; option<num_options; ++option) {
        options[option] = option;
    }
    hole_options.push_back(options);
    return hole;
}

void Family::holeSetOption(uint64_t hole, uint64_t option) {
    hole_options[hole].clear();
    hole_options[hole].push_back(option);
    hole_options_mask[hole].clear();
    hole_options_mask[hole].set(option);
}
void Family::holeSetOptions(uint64_t hole, std::vector<uint64_t> const& options) {
    hole_options_mask[hole].clear();
    for(uint64_t option: options) {
        hole_options_mask[hole].set(option);
    }
    hole_options[hole].clear();
    for(uint64_t option: hole_options_mask[hole]) {
        hole_options[hole].push_back(option);
    }
}
void Family::holeSetOptions(uint64_t hole, BitVector const& options) {
    hole_options[hole].clear();
    for(uint64_t option: options) {
        hole_options[hole].push_back(option);
    }
    hole_options_mask[hole] = options;
}

std::vector<uint64_t> const& Family::holeOptions(uint64_t hole) const {
    return hole_options[hole];
}

BitVector const& Family::holeOptionsMask(uint64_t hole) const {
    return hole_options_mask[hole];
}

uint64_t Family::holeNumOptions(uint64_t hole) const {
    return hole_options[hole].size();
}

uint64_t Family::holeNumOptionsTotal(uint64_t hole) const {
    return hole_options_mask[hole].size();
}

/*uint128_t Family::size() const {
    uint128_t size = 1;
    for(uint64_t hole = 0; hole < numHoles(); ++hole) {
        size *= holeNumOptions(hole);
    }
    return size;
}*/

bool Family::isAssignment() const {
    for(uint64_t hole = 0; hole < numHoles(); ++hole) {
        if(holeNumOptions(hole) > 1) {
            return false;
        }
    }
    return true;
}

bool Family::holeContains(uint64_t hole, uint64_t option) const {
    return hole_options_mask[hole][option];
}


bool Family::isSubsetOf(Family const& other) const {
    for(uint64_t hole = 0; hole < numHoles(); ++hole) {
        if(not hole_options_mask[hole].isSubsetOf(other.holeOptionsMask(hole))) {
            return false;
        }
    }
    return true;
}

bool Family::includesAssignment(std::vector<uint64_t> const& hole_to_option) const {
    for(uint64_t hole = 0; hole < numHoles(); ++hole) {
        if(not hole_options_mask[hole][hole_to_option[hole]]) {
            return false;
        }
    }
    return true;
}

bool Family::includesAssignment(std::map<uint64_t,uint64_t> const& hole_to_option) const {
    for(auto const& [hole,option]: hole_to_option) {
        if(not hole_options_mask[hole][option]) {
            return false;
        }
    }
    return true;
}

bool Family::includesAssignment(std::vector<std::pair<uint64_t,uint64_t>> const& hole_to_option) const {
    for(auto const& [hole,option]: hole_to_option) {
    if(not hole_options_mask[hole][option]) {
            return false;
        }
    }
    return true;   
}

std::vector<BitVector>::const_iterator Family::begin() const {
    return hole_options_mask.begin();
}

std::vector<BitVector>::const_iterator Family::end() const {
    return hole_options_mask.end();
}


void Family::setChoices(BitVector const& choices) {
    this->choices = BitVector(choices);
}

void Family::setChoices(BitVector && choices) {
    this->choices = choices;
}

BitVector const& Family::getChoices() const {
    return choices;
}

}