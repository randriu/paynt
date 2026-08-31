#include "src/synthesis/synthesis.h"
#include "SubPomdpBuilder.h"

#include <queue>

#include "src/synthesis/translation/componentTranslations.h"
#include "src/synthesis/translation/choiceTransformation.h"
#include <storm/adapters/RationalNumberAdapter.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/BitVector.h>
#include <storm/models/sparse/Dtmc.h>
#include <storm/builder/RewardModelBuilder.h>
#include <storm/builder/RewardModelInformation.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/utility/macros.h>

namespace synthesis {
template <typename ValueType>
std::shared_ptr<storm::models::sparse::Mdp<ValueType>> createMdpFromVectorMatrix(storm::models::sparse::Mdp<ValueType> mdp, std::vector<std::vector<std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>>> vectorMatrix) {

    uint64_t row_count = 0;
    for (const auto& v : vectorMatrix) {
        row_count += v.size();
    }

    storm::storage::sparse::ModelComponents<ValueType> components;

    storm::storage::SparseMatrixBuilder<ValueType> builder(
        row_count, vectorMatrix.size(), 0, false, true, vectorMatrix.size()
    );
    
    uint64_t current_row = 0;
    for (const auto& rowGroupVector : vectorMatrix) {
        builder.newRowGroup(current_row);
        for (const auto& row : rowGroupVector) {
            for (const auto& entry : row) {
                builder.addNextValue(current_row, entry.getColumn(), entry.getValue());
            }
            ++current_row;
        }
    }

    components.transitionMatrix =  builder.build();

    components.stateLabeling = mdp.getStateLabeling();

    return std::make_shared<storm::models::sparse::Mdp<ValueType>>(std::move(components));
}

template <typename ValueType>
std::vector<std::vector<std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>>> getVectorFromMatrix(storm::storage::SparseMatrix<ValueType> matrix) {

    uint64_t stateCount = matrix.getRowGroupCount(); 
    std::vector<std::vector<std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>>> resultVector(stateCount);
    const auto rowGroupIndices = matrix.getRowGroupIndices();

    for (uint64_t state = 0; state < stateCount; ++state) {
        resultVector[state].resize(rowGroupIndices[state+1] - rowGroupIndices[state]);
        uint64_t startRow = rowGroupIndices[state];
        for (uint64_t row = rowGroupIndices[state]; row < rowGroupIndices[state+1]; row++) {
            auto rowData = matrix.getRow(row);
            resultVector[state][row - startRow].resize(rowData.getNumberOfEntries());
            for (const auto &entry : rowData) {
                resultVector[state][row - startRow].push_back(entry);
            }
        }
    }

    return resultVector;
}

template <typename ValueType>
std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>> createCombinationOfRows(std::vector<std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>> rows, std::vector<ValueType> distribution) {
    // Use a map to accumulate values for each column
    std::map<uint_fast64_t, ValueType> columnSums;
    for (size_t i = 0; i < rows.size(); ++i) {
        for (const auto& entry : rows[i]) {
            columnSums[entry.getColumn()] += entry.getValue() * distribution[i];
        }
    }

    std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>> combinedEntries;
    for (const auto& [column, value] : columnSums) {
        combinedEntries.emplace_back(column, value);
    }

    return combinedEntries;
}

template <typename ValueType>
std::pair<std::shared_ptr<storm::models::sparse::Mdp<ValueType>>, std::vector<std::vector<int64_t>>> createSlidingWindowMemoryMdp(storm::models::sparse::Mdp<ValueType> originalMdp, uint64_t windowMemorySize) {
    STORM_LOG_ASSERT(windowMemorySize > 0, "Sliding window memory size must be greater than 0");

    // compute state space
    std::vector<int64_t> initialStateWindow(windowMemorySize, -1);
    uint64_t initialState = *(originalMdp.getInitialStates().begin());
    initialStateWindow[0] = static_cast<int64_t>(initialState);
    std::queue<std::vector<int64_t>> stateQueue;
    stateQueue.emplace(initialStateWindow);
    std::vector<std::vector<int64_t>> newStates;
    newStates.push_back(initialStateWindow);

    const auto originalMatrix = originalMdp.getTransitionMatrix();
    const auto rowGroupIndices = originalMatrix.getRowGroupIndices();

    //bfs
    while (!stateQueue.empty()) {
        std::vector<int64_t> currentStateWindow = stateQueue.front();
        stateQueue.pop();

        uint64_t originalState = static_cast<uint64_t>(currentStateWindow[0]);
        for (uint64_t row = rowGroupIndices[originalState]; row < rowGroupIndices[originalState+1]; row++) {
            auto rowData = originalMatrix.getRow(row);
            for (const auto &entry : rowData) {
                std::vector<int64_t> newStateWindow = currentStateWindow;
                // Shift all elements right by one position
                for (size_t i = newStateWindow.size() - 1; i > 0; --i) {
                    newStateWindow[i] = newStateWindow[i - 1];
                }
                // Set the first element to the new state
                newStateWindow[0] = static_cast<int64_t>(entry.getColumn());
                
                if (std::find(newStates.begin(), newStates.end(), newStateWindow) == newStates.end()) {
                    newStates.push_back(newStateWindow);
                    stateQueue.push(newStateWindow);
                }
            }
        }
    }

    uint64_t row_count = 0;
    for (const auto& v : newStates) {
        uint64_t originalState = static_cast<uint64_t>(v[0]);
        row_count += rowGroupIndices[originalState+1] - rowGroupIndices[originalState];
    }

    storm::storage::SparseMatrixBuilder<ValueType> builder(
        row_count, newStates.size(), 0, false, true, newStates.size()
    );

    std::map<std::string, storm::storage::BitVector> stateLabelsBitVectors;

    storm::models::sparse::StateLabeling labeling(newStates.size());
    for (auto const& label : originalMdp.getStateLabeling().getLabels()) {
        stateLabelsBitVectors[label] = storm::storage::BitVector(newStates.size(), false);
    }

    uint64_t current_row = 0;
    uint64_t stateIndex = 0;
    for (const auto& stateWindow : newStates) {
        uint64_t originalState = static_cast<uint64_t>(stateWindow[0]);
        builder.newRowGroup(current_row);
        for (uint64_t row = rowGroupIndices[originalState]; row < rowGroupIndices[originalState+1]; row++) {
            auto rowData = originalMatrix.getRow(row);
            for (const auto &entry : rowData) {
                std::vector<int64_t> newStateWindow = stateWindow;
                // Shift all elements right by one position
                for (size_t i = newStateWindow.size() - 1; i > 0; --i) {
                    newStateWindow[i] = newStateWindow[i - 1];
                }
                // Set the first element to the new state
                newStateWindow[0] = static_cast<int64_t>(entry.getColumn());
                uint64_t newStateIndex = std::distance(newStates.begin(), std::find(newStates.begin(), newStates.end(), newStateWindow));
                builder.addNextValue(current_row, newStateIndex, entry.getValue());
            }
            current_row++;
        }

        for (auto const& label : originalMdp.getStateLabeling().getLabels()) {
            if (originalMdp.getStateLabeling().getLabelsOfState(originalState).count(label) > 0) {
                stateLabelsBitVectors[label].set(stateIndex, true);
            }
        }
        stateIndex++;
    }

    storm::storage::sparse::ModelComponents<ValueType> components;

    components.transitionMatrix =  builder.build();

    for (auto const& label : originalMdp.getStateLabeling().getLabels()) {
        if (label == "init") {
            storm::storage::BitVector initLabeling(newStates.size(), false);
            initLabeling.set(0, true); // only the initial state gets the init label
            labeling.addLabel(label, std::move(initLabeling));
            continue;
        }
        labeling.addLabel(label, std::move(stateLabelsBitVectors[label]));
    }

    components.stateLabeling = labeling;

    auto newMdp = std::make_shared<storm::models::sparse::Mdp<ValueType>>(std::move(components));

    return {newMdp, newStates};
}


template <typename ValueType>
std::shared_ptr<storm::models::sparse::Dtmc<ValueType>> applyRandomizedScheduler(
    storm::models::sparse::Mdp<ValueType> const& mdp,
    std::vector<std::vector<ValueType>> actionDistributions) {

    const auto originalMatrix = mdp.getTransitionMatrix();
    const auto rowGroupIndices = originalMatrix.getRowGroupIndices();

    uint64_t num_states = originalMatrix.getRowGroupCount();

    storm::storage::SparseMatrixBuilder<ValueType> builder(
        num_states, num_states, 0, false, false, num_states
    );

    storm::builder::RewardModelInformation rewardModelInfo("rews", false, true, false);
    storm::builder::RewardModelBuilder<ValueType> rewardBuilder(rewardModelInfo);

    auto const& originalRewardModel = mdp.getRewardModel("rews");

    for (uint64_t state = 0; state < num_states; ++state) {
        uint64_t actionCount = rowGroupIndices[state+1] - rowGroupIndices[state];
        STORM_LOG_ASSERT(actionDistributions[state].size() == actionCount, "Action distribution size must match the number of actions for the state");

        // Combine the rows according to the action distribution
        std::map<uint64_t, ValueType> combinedRow;
        ValueType combinedReward = storm::utility::zero<ValueType>();
        for (uint64_t action = 0; action < actionCount; ++action) {
            ValueType actionProb = actionDistributions[state][action];
            auto rowData = originalMatrix.getRow(rowGroupIndices[state] + action);
            for (const auto &entry : rowData) {
                combinedRow[entry.getColumn()] += entry.getValue() * actionProb;
            }
            combinedReward += actionProb * originalRewardModel.getStateActionReward(rowGroupIndices[state]+action);
        }

        for (const auto& [column, value] : combinedRow) {
            builder.addNextValue(state, column, value);
        }
        rewardBuilder.addStateActionReward(combinedReward);
    }

    storm::storage::sparse::ModelComponents<ValueType> components;
    components.transitionMatrix = builder.build();
    components.stateLabeling = mdp.getStateLabeling();
    components.rewardModels.emplace(rewardBuilder.getName(), rewardBuilder.build(num_states, num_states, num_states));

    return std::make_shared<storm::models::sparse::Dtmc<ValueType>>(std::move(components));
}

template <typename ValueType>
std::shared_ptr<storm::models::sparse::Dtmc<ValueType>> applyRandomizedSchedulerFromTree(
    storm::models::sparse::Mdp<ValueType> const& mdp,
    std::vector<std::vector<ValueType>> actionDistributions,
    std::map<uint64_t, uint64_t> nodeToStateMapping,
    std::vector<std::map<uint64_t, uint64_t>> nodeToSuccessorToStatesMap) {

    const auto originalMatrix = mdp.getTransitionMatrix();
    const auto rowGroupIndices = originalMatrix.getRowGroupIndices();

    uint64_t num_states = nodeToStateMapping.size() + originalMatrix.getRowGroupCount();

    storm::storage::SparseMatrixBuilder<ValueType> builder(
        num_states, num_states, 0, false, false, num_states
    );

    storm::builder::RewardModelInformation rewardModelInfo("rews", false, true, false);
    storm::builder::RewardModelBuilder<ValueType> rewardBuilder(rewardModelInfo);

    auto const& originalRewardModel = mdp.getRewardModel("rews");

    for (uint64_t state = 0; state < num_states; ++state) {
        uint64_t originalState;
        bool treeNodeState = false;
        if (state < nodeToStateMapping.size()) {
            originalState = nodeToStateMapping.at(state);
            treeNodeState = true;
        } else {
            originalState = state - nodeToStateMapping.size();
        }

        uint64_t actionCount = rowGroupIndices[originalState+1] - rowGroupIndices[originalState];
        STORM_LOG_ASSERT(actionDistributions[state].size() == actionCount, "Action distribution size must match the number of actions for the state");

        // Combine the rows according to the action distribution
        std::map<uint64_t, ValueType> combinedRow;
        ValueType combinedReward = storm::utility::zero<ValueType>();
        for (uint64_t action = 0; action < actionCount; ++action) {
            ValueType actionProb = actionDistributions[state][action];
            auto rowData = originalMatrix.getRow(rowGroupIndices[originalState] + action);
            for (const auto &entry : rowData) {
                combinedRow[entry.getColumn()] += entry.getValue() * actionProb;
            }
            combinedReward += actionProb * originalRewardModel.getStateActionReward(rowGroupIndices[originalState]+action);
        }

        for (const auto& [column, value] : combinedRow) {
            uint64_t mappedColumn;
            if (treeNodeState && nodeToSuccessorToStatesMap[state].count(column) > 0) {
                mappedColumn = nodeToSuccessorToStatesMap[state].at(column);
            } else {
                mappedColumn = column + nodeToStateMapping.size();
            }
            builder.addNextValue(state, mappedColumn, value);
        }
        rewardBuilder.addStateActionReward(combinedReward);
    }

    storm::storage::sparse::ModelComponents<ValueType> components;
    components.transitionMatrix = builder.build();
    components.rewardModels.emplace(rewardBuilder.getName(), rewardBuilder.build(num_states, num_states, num_states));

    storm::models::sparse::StateLabeling labeling(num_states);
    storm::storage::BitVector initLabeling(num_states, false);
    initLabeling.set(0, true); // only the initial state gets the init label

    std::map<std::string, storm::storage::BitVector> stateLabelsBitVectors;
    for (auto const& label : mdp.getStateLabeling().getLabels()) {
        if (label == "init") {
            stateLabelsBitVectors["init"] = initLabeling;
            continue;
        }
        stateLabelsBitVectors[label] = storm::storage::BitVector(num_states, false);
    }

    for (uint64_t state = 0; state < num_states; ++state) {
        
        uint64_t originalState;
        bool treeNodeState = false;
        if (state < nodeToStateMapping.size()) {
            originalState = nodeToStateMapping.at(state);
            treeNodeState = true;
        } else {
            originalState = state - nodeToStateMapping.size();
        }
        
        for (auto const& label : mdp.getStateLabeling().getLabelsOfState(originalState)) {
            if (label == "init") {
                continue;
            }
            if (stateLabelsBitVectors.count(label) > 0) {
                stateLabelsBitVectors[label].set(state, true);
            }
        }
    }

    for (auto const& label : mdp.getStateLabeling().getLabels()) {
        labeling.addLabel(label, std::move(stateLabelsBitVectors[label]));
    }

    components.stateLabeling = labeling;

    return std::make_shared<storm::models::sparse::Dtmc<ValueType>>(std::move(components));
}


template <typename ValueType>
std::shared_ptr<storm::models::sparse::Dtmc<ValueType>> createDtmcFromVectorMatrixWithRewards(storm::models::sparse::Mdp<ValueType> mdp, std::vector<std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>> vectorMatrix, std::vector<ValueType> rewards) {

    const auto unfoldedMatrix = mdp.getTransitionMatrix();
    uint64_t num_states = unfoldedMatrix.getRowGroupCount();

    storm::storage::sparse::ModelComponents<ValueType> components;

    storm::storage::SparseMatrixBuilder<ValueType> builder(
        num_states, num_states, 0, false, false, num_states
    );

    storm::builder::RewardModelInformation rewardModelInfo("rews", false, true, false);
    storm::builder::RewardModelBuilder<ValueType> rewardBuilder(rewardModelInfo);
    
    uint64_t current_row = 0;
    for (const auto& row : vectorMatrix) {
        for (const auto& entry : row) {
            builder.addNextValue(current_row, entry.getColumn(), entry.getValue());
        }
        rewardBuilder.addStateActionReward(rewards[current_row]);
        ++current_row;
    }

    components.transitionMatrix =  builder.build();
    components.rewardModels.emplace(rewardBuilder.getName(), rewardBuilder.build(num_states, num_states, num_states));

    components.stateLabeling = mdp.getStateLabeling();

    return std::make_shared<storm::models::sparse::Dtmc<ValueType>>(std::move(components));
}


template <typename ValueType>
std::pair<std::shared_ptr<storm::models::sparse::Dtmc<ValueType>>, std::vector<std::pair<uint64_t, uint64_t>>> unfoldDtmcWithStepBound(storm::models::sparse::Dtmc<ValueType> dtmc, uint64_t stepBound) {


    // compute state space
    std::pair<uint64_t, uint64_t> initialStateAndStep = {*(dtmc.getInitialStates().begin()), 0};

    std::queue<std::pair<uint64_t, uint64_t>> stateQueue;
    stateQueue.emplace(initialStateAndStep);
    std::vector<std::pair<uint64_t, uint64_t>> newStates;
    newStates.push_back(initialStateAndStep);

    const auto originalMatrix = dtmc.getTransitionMatrix();
    const auto originalStateCount = originalMatrix.getRowCount();

    //bfs
    while (!stateQueue.empty()) {
        std::pair<uint64_t, uint64_t> currentStateAndStep = stateQueue.front();
        stateQueue.pop();

        uint64_t originalState = currentStateAndStep.first;
        uint64_t currentStep = currentStateAndStep.second;
        if (currentStep >= stepBound) {
            continue;
        }

        auto rowData = originalMatrix.getRow(originalState);
        for (const auto &entry : rowData) {
            std::pair<uint64_t, uint64_t> newStateAndStep = {static_cast<uint64_t>(entry.getColumn()), currentStep + 1};
                
            if (std::find(newStates.begin(), newStates.end(), newStateAndStep) == newStates.end()) {
                newStates.push_back(newStateAndStep);
                stateQueue.push(newStateAndStep);
            }
        }
    }

    uint64_t num_states = newStates.size() + 1; // +1 for absorbing state

    storm::storage::sparse::ModelComponents<ValueType> components;

    storm::storage::SparseMatrixBuilder<ValueType> builder(
        num_states, num_states, 0, false, false, num_states
    );

    for (uint64_t new_state = 0; new_state < num_states; ++new_state) {
        if (new_state == num_states - 1) {
            // absorbing state
            builder.addNextValue(new_state, new_state, storm::utility::one<ValueType>());
            continue;
        }
        uint64_t original_state = newStates[new_state].first;
        uint64_t step = newStates[new_state].second;

        auto rowData = originalMatrix.getRow(original_state);
        for (const auto &entry : rowData) {
            uint64_t target_new_state;
            if (step + 1 >= stepBound) {
                target_new_state = num_states - 1; // go to absorbing state
            } else {
                std::pair<uint64_t, uint64_t> targetStateAndStep = {entry.getColumn(), step + 1};
                target_new_state = std::distance(newStates.begin(), std::find(newStates.begin(), newStates.end(), targetStateAndStep));
            }
            builder.addNextValue(new_state, target_new_state, entry.getValue());
        }
    }

    components.transitionMatrix =  builder.build();

    storm::models::sparse::StateLabeling labeling(num_states);
    storm::storage::BitVector initLabeling(num_states, false);
    initLabeling.set(0, true); // only the initial state gets the init label

    storm::storage::BitVector stepSinkLabeling(num_states, false);
    stepSinkLabeling.set(num_states - 1, true);

    std::map<std::string, storm::storage::BitVector> stateLabelsBitVectors;
    for (auto const& label : dtmc.getStateLabeling().getLabels()) {
        if (label == "init") {
            stateLabelsBitVectors["init"] = initLabeling;
            continue;
        }
        stateLabelsBitVectors[label] = storm::storage::BitVector(num_states, false);
    }

    stateLabelsBitVectors["step_sink"] = stepSinkLabeling;

    for (uint64_t state = 0; state < num_states - 1; ++state) {
        
        uint64_t originalState = newStates[state].first;
        
        for (auto const& label : dtmc.getStateLabeling().getLabelsOfState(originalState)) {
            if (label == "init") {
                continue;
            }
            if (stateLabelsBitVectors.count(label) > 0) {
                stateLabelsBitVectors[label].set(state, true);
            }
        }
    }

    for (auto const& label : dtmc.getStateLabeling().getLabels()) {
        labeling.addLabel(label, std::move(stateLabelsBitVectors[label]));
    }

    components.stateLabeling = labeling;

    return std::make_pair(std::make_shared<storm::models::sparse::Dtmc<ValueType>>(std::move(components)), newStates);
}


}  // namespace synthesis

template <typename ValueType>
void bindings_translation_vt(py::module& m, std::string const& vtSuffix) {

    m.def(("computeChoiceDestinations" + vtSuffix).c_str(), &synthesis::computeChoiceDestinations<ValueType>);
    m.def(("addMissingChoiceLabels" + vtSuffix).c_str(), &synthesis::addMissingChoiceLabelsModel<ValueType>);
    m.def(("assertChoiceLabelingIsCanonic" + vtSuffix).c_str(), &synthesis::assertChoiceLabelingIsCanonic);
    m.def(("extractActionLabels" + vtSuffix).c_str(), &synthesis::extractActionLabels<ValueType>);
    m.def(("enableAllActions" + vtSuffix).c_str(), py::overload_cast<storm::models::sparse::Model<ValueType> const&>(&synthesis::enableAllActions<ValueType>));
    m.def(("restoreActionsInAbsorbingStates" + vtSuffix).c_str(), &synthesis::restoreActionsInAbsorbingStates<ValueType>);
    m.def(("addDontCareAction" + vtSuffix).c_str(), &synthesis::addDontCareAction<ValueType>);
    m.def(("createModelUnion" + vtSuffix).c_str(), &synthesis::createModelUnion<ValueType>);

    m.def(("createMdpFromVectorMatrix" + vtSuffix).c_str(), &synthesis::createMdpFromVectorMatrix<ValueType>);
    m.def(("getVectorFromMatrix" + vtSuffix).c_str(), &synthesis::getVectorFromMatrix<ValueType>);
    m.def(("createCombinationOfRows" + vtSuffix).c_str(), &synthesis::createCombinationOfRows<ValueType>);
    m.def(("createSlidingWindowMemoryMdp" + vtSuffix).c_str(), &synthesis::createSlidingWindowMemoryMdp<ValueType>);
    m.def(("applyRandomizedScheduler" + vtSuffix).c_str(), &synthesis::applyRandomizedScheduler<ValueType>);
    m.def(("applyRandomizedSchedulerFromTree" + vtSuffix).c_str(), &synthesis::applyRandomizedSchedulerFromTree<ValueType>);
    m.def(("createDtmcFromVectorMatrixWithRewards" + vtSuffix).c_str(), &synthesis::createDtmcFromVectorMatrixWithRewards<ValueType>);
    m.def(("unfoldDtmcWithStepBound" + vtSuffix).c_str(), &synthesis::unfoldDtmcWithStepBound<ValueType>);

    m.def(("get_matrix_rows" + vtSuffix).c_str(), [](storm::storage::SparseMatrix<ValueType>& matrix, std::vector<typename storm::storage::SparseMatrix<ValueType>::index_type> rows) {
        std::vector<typename storm::storage::SparseMatrix<ValueType>::rows> result;
        for (const auto& row : rows) {
            result.push_back(matrix.getRows(row, row+1));
        }
        return result;
    }, py::arg("matrix"), py::arg("rows"), "Get multiple rows from a sparse matrix");

    py::classh<synthesis::SubPomdpBuilder<ValueType>>(m, ("SubPomdpBuilder" + vtSuffix).c_str())
        .def(py::init<storm::models::sparse::Pomdp<ValueType> const&>())
        .def("start_from_belief", &synthesis::SubPomdpBuilder<ValueType>::startFromBelief)
        .def_property_readonly("state_sub_to_full", [](synthesis::SubPomdpBuilder<ValueType>& b) {return b.state_sub_to_full;} )
        ;
}

void bindings_translation(py::module& m) {
    bindings_translation_vt<double>(m, "");
    bindings_translation_vt<storm::RationalNumber>(m, "Exact");
}
