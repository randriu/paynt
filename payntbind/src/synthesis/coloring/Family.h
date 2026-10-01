#pragma once

#include <storm/storage/BitVector.h>

#include <cstdint>
#include <vector>
#include <map>

namespace synthesis {

using BitVector = storm::storage::BitVector;
// typedef unsigned __int128 uint128_t;

class Family {
public:
    
    Family() {};
    Family(Family const& other);

    uint64_t numHoles() const;
    uint64_t addHole(uint64_t num_options);

    void holeSetOption(uint64_t hole, uint64_t option);
    void holeSetOptions(uint64_t hole, std::vector<uint64_t> const& options);
    void holeSetOptions(uint64_t hole, BitVector const& options);
    // void holeSetOptions(uint64_t hole, BitVector&& options);

    std::vector<uint64_t> const& holeOptions(uint64_t hole) const;
    BitVector const& holeOptionsMask(uint64_t hole) const;

    uint64_t holeNumOptions(uint64_t hole) const;
    uint64_t holeNumOptionsTotal(uint64_t hole) const;
    // uint128_t size() const;
    bool isAssignment() const;

    bool holeContains(uint64_t hole, uint64_t option) const;
    bool isSubsetOf(Family const& other) const;
    bool includesAssignment(std::vector<uint64_t> const& hole_to_option) const;
    bool includesAssignment(std::map<uint64_t,uint64_t> const& hole_to_option) const;
    bool includesAssignment(std::vector<std::pair<uint64_t,uint64_t>> const& hole_to_option) const;

    // iterator over hole options
    std::vector<BitVector>::const_iterator begin() const;
    std::vector<BitVector>::const_iterator end() const;

    // choice operations
    void setChoices(BitVector const& choices);
    void setChoices(BitVector&& choices);
    BitVector const& getChoices() const;

protected:

    /** For each hole, a list of available options. */
    std::vector<std::vector<uint64_t>> hole_options;
    /** For each hole, a mastk of available options. */
    std::vector<BitVector> hole_options_mask;

    /** Whether choices have been set for this family. */
    bool choices_set = false;
    /** Bitvector of choices relevant to this family. */
    BitVector choices;
};

}