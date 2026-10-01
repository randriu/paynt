#pragma once

#include "src/synthesis/coloring/Family.h"

#include <storm/storage/BitVector.h>

#include <cstdint>
#include <vector>
#include <memory>

namespace synthesis {

using BitVector = storm::storage::BitVector;


class Coloring {
public:
    
    Coloring(
        Family const& family, std::vector<uint64_t> const& row_groups,
        std::vector<std::vector<std::pair<uint64_t,uint64_t>>> choice_to_assignment
    );

    /** Get choice-to-assignment mapping. */
    std::vector<std::vector<std::pair<uint64_t,uint64_t>>> const& getChoiceToAssignment() const;
    /** Get a mapping from states to holes involved in its choices. */
    std::vector<BitVector> const& getStateToHoles() const;
    
    /** Get a mask of choices compatible with the family. */
    BitVector selectCompatibleChoices(Family const& subfamily) const;
    /**
     * Get a mask of choices compatible with the family, restricting the *colored-choice* search to
     * base_choices. Uncolored choices (no hole/option assignment at all) are always included regardless of
     * base_choices -- they carry nothing for any subfamily to exclude them on, so base_choices restricting
     * them too would be unsound, not just redundant (confirmed by trying it: every state's uncolored
     * fallback choices are load-bearing for basic model well-formedness, and excluding any of them produces
     * deadlock states in the constructed submodel). Sound for the colored portion whenever subfamily is a
     * narrowing of the family that produced base_choices (i.e. every hole's allowed options in subfamily are
     * a subset of what produced base_choices): a colored choice compatible with the narrower subfamily must
     * also be compatible with the wider family, so any colored choice already excluded from base_choices is
     * guaranteed to stay excluded and can be skipped, without changing the result.
     */
    BitVector selectCompatibleChoices(Family const& subfamily, BitVector const& base_choices) const;
    /** For each hole, collect options (colors) involved in any of the given choices. */
    std::vector<std::vector<uint64_t>> collectHoleOptions(BitVector const& choices) const;
    
protected:

    /** Reference to the unrefined family. */
    Family family;
    /** For each choice, a list of hole-option pairs (colors). */
    const std::vector<std::vector<std::pair<uint64_t,uint64_t>>> choice_to_assignment;

    /** Number of choices in the quotient. */
    const uint64_t numChoices() const;
    
    /** For each state, identification of holes associated with its choices. */
    std::vector<BitVector> choice_to_holes;
    /** For each state, identification of holes associated with its choices. */
    std::vector<BitVector> state_to_holes;

    /** Choices not labeled by any hole. */
    BitVector uncolored_choices;
    /** Choices labeled by some hole. */
    BitVector colored_choices;

    /** For each hole, collect options (colors) involved in any of the given choices. */
    std::vector<BitVector> collectHoleOptionsMask(BitVector const& choices) const;
};

}