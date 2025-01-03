"""Metrics for the belief state of the agent."""

import numpy as np

def negative_log_likelihood(
    state, belief_state, use_importance_weight=True, verbose=False
) -> float:
  """Computes the negative log likelihood of a state given agent's belief state.
  """
  nll = 0.
  def find_attribute_prob(attributes, attribute, value):
    for a in attributes:
      if a.name == attribute.name:
        for v in a.value:
          if v.name == value.name:
            if v.probability < 1e-10:
              print(f'probability is too small for {attribute.name}')
              return 1e-10
            return v.probability
    return 1e-10

  def find_prob(belief, entity, attribute, value):
    if attribute is None:
      for e in belief.all_entities:
        if e.name == entity.name:
          if e.probability < 1e-10:
            print(f'e.probability is too small for entity: {entity.name}')
            return 1e-10
          return e.probability
      return 1e-10
    for e in belief.all_entities:
      if e.name == entity.name:
        return find_attribute_prob(e.attributes, attribute, value)
    return 1e-10

  for entity in state.all_entities:
    prob = find_prob(belief_state, entity, None, None)
    nll -= np.log(prob) * (
        entity.importance_score if use_importance_weight else 1.0
    )
    if verbose:
      print(
          f'entity: {entity.name}, importance_score: {entity.importance_score},'
          f' prob: {prob}'
      )
    for attribute in entity.attributes:
      value = attribute.value[0]
      assert np.isclose(
          value.probability, 1.0
      ), f'attribute: {attribute}, prob: {value.probability}'
      # find the corresponding prob in belief
      prob = find_prob(belief_state, entity, attribute, value)
      nll -= np.log(prob) * (
          entity.importance_score * attribute.importance_score
          if use_importance_weight
          else 1.0
      )
      if verbose:
        print(
            f'attribute: {attribute.name}, importance_score:'
            f' {attribute.importance_score}, prob: {prob}'
        )
  for relation in state.all_relations:
    value = relation.value[0]
    assert np.isclose(
        value.probability, 1.0
    ), f'relation: {relation}, prob: {value.probability}'
    # find the corresponding prob in belief
    prob = find_attribute_prob(belief_state.all_relations, relation, value)
    nll -= np.log(prob) * (
        relation.importance_score if use_importance_weight else 1.0
    )
    if verbose:
      print(
          f'relation: {relation.name}, importance_score:'
          f' {relation.importance_score}, prob: {prob}'
      )
  return nll
