#include "JaniChoices.h"
#include "src/synthesis/translation/choiceTransformation.h"

#include <storm/models/sparse/Mdp.h>
#include <storm/storage/jani/Model.h>
#include <storm/storage/sparse/JaniChoiceOrigins.h>
#include <storm/storage/sparse/ModelComponents.h>

namespace synthesis {
    
    template<typename ValueType>
    storm::models::sparse::ChoiceLabeling reconstructChoiceLabelsFromJani(
        storm::models::sparse::Model<ValueType> const& model
    ) {
        uint64_t num_choices = model.getNumberOfChoices();
        auto const& co = model.getChoiceOrigins()->asJaniChoiceOrigins();
        auto const& jani = co.getModel();

        storm::models::sparse::ChoiceLabeling choice_labeling(num_choices);
        for (auto const& action : jani.getActions()) {
            choice_labeling.addLabel(action.getName(), storm::storage::BitVector(num_choices,false));
        }
        
        for(uint64_t choice = 0; choice < num_choices; choice++) {
            for(auto const& aut_edge: co.getEdgeIndexSet(choice)) {
                auto [aut_index,edge_index] = jani.decodeAutomatonAndEdgeIndices(aut_edge);
                auto action_index = jani.getAutomaton(aut_index).getEdge(edge_index).getActionIndex();
                choice_labeling.addLabelToChoice(jani.getAction(action_index).getName(), choice);
            }
        }

        removeUnusedLabels(choice_labeling);
        return choice_labeling;
    }

    template<typename ValueType>
    std::shared_ptr<storm::models::sparse::Model<ValueType>> addChoiceLabelsFromJani(storm::models::sparse::Model<ValueType> const& model) {
        storm::storage::sparse::ModelComponents<ValueType> components;
        components.transitionMatrix = model.getTransitionMatrix();
        components.stateLabeling = model.getStateLabeling();
        if(model.hasStateValuations()) {
            components.stateValuations = model.getStateValuations();
        }
        components.choiceLabeling = reconstructChoiceLabelsFromJani(model);
        addMissingChoiceLabelsLabeling(model,components.choiceLabeling.value());
        components.rewardModels = model.getRewardModels();
        return std::make_shared<storm::models::sparse::Mdp<ValueType>>(std::move(components));
    }
    

    template std::shared_ptr<storm::models::sparse::Model<double>> addChoiceLabelsFromJani(storm::models::sparse::Model<double> const& model);
    template std::shared_ptr<storm::models::sparse::Model<storm::RationalNumber>> addChoiceLabelsFromJani(storm::models::sparse::Model<storm::RationalNumber> const& model);
}
