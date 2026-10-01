#pragma once

#include <storm/adapters/RationalNumberAdapter.h>
#include <storm/models/sparse/Model.h>

namespace synthesis {

    /**
     * Given model with Jani choice origins, reconstuct the corresponding choice labels.
     */
    template<typename ValueType>
    std::shared_ptr<storm::models::sparse::Model<ValueType>> addChoiceLabelsFromJani(
        storm::models::sparse::Model<ValueType> const& model
    );
}