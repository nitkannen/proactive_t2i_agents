"""Utils for belief state."""

import copy
import random
import re
import numpy as np

from agent import agent_classes as ac
from agent import prompts
from agent import data_classes as dc
from agent import llm_parser as parser
from agent import utility_classes as utils


def is_confident(
    belief_state, confidence_threshold=0.9, importance_threshold=0.1
):
  """Check if the agent is confident with respect to some thresholds.

  Args:
    belief_state: The belief state of the agent to be checked.
    confidence_threshold: The threshold for the confidence of the agent. The
      agent is confident if the probability of the attribute candidate is less
      than or equal to 1 - confidence_threshold or greater than or equal to
      confidence_threshold. For example, if the confidence_threshold is 0.9, the
      agent is confident if the probability of the attribute candidate is less
      than or equal to 0.1 or greater than or equal to 0.9.
    importance_threshold: The threshold for the importance of the entity,
      attribute or relation. We only consider entities, attributes and relations
      with importance score greater than or equal to the importance_threshold.
      If the specific item is not important, we do not check the confidence of
      the item.

  Returns:
    True if the agent is confident, False otherwise.
  """
  assert confidence_threshold >= 0.5, 'confidence_threshold must be >= 0.5.'
  assert importance_threshold >= 0.0, 'importance_threshold must be >= 0.0.'

  def not_confident(probability):
    return (
        probability > 1 - confidence_threshold
        and probability < confidence_threshold
    )

  for entity in belief_state.all_entities:
    if entity.importance_score < importance_threshold:
      continue
    for attribute in entity.attributes:
      if attribute.importance_score < importance_threshold:
        continue
      for candidate in attribute.value:
        if not_confident(candidate.probability):
          return False
  for relation in belief_state.all_relations:
    if relation.importance_score < importance_threshold:
      continue
    for candidate in relation.value:
      if not_confident(candidate.probability):
        return False
  return True


def most_likely_deterministic_state(
    belief_state,
    remove_uncertain_entities=False,
    remove_uncertain_attributes=False,
    prob_threshold=0.7,
):
  """Obtain the most likely deterministic state from a belief state.

  The most likely deterministic state is a belief state where all the attributes
  and relations have only one candidate with probability 1. We set all
  attributes and relations to be the most likely candidate. This function may
  introduce conflicts between attributes and relations. Use this function only
  when the agent is confident.

  Args:
    belief_state: The belief state to be processed.
    remove_uncertain_entities: If True, remove entities with probability less
      than prob_threshold.
    remove_uncertain_attributes: If True, remove attributes with max probability
      less than prob_threshold.
    prob_threshold: The threshold for the probability of the attribute
      candidate. The attribute candidate with probability less than
      prob_threshold will be ignored.

  Returns:
    The most likely deterministic state.
  """

  def most_likely_candidate(candidates):
    candidate = max(candidates, key=lambda x: x.probability)
    original_prob = candidate.probability
    candidate.probability = 1.0
    return candidate, original_prob

  new_state = copy.deepcopy(belief_state)
  all_entities = []
  for entity in new_state.all_entities:
    if entity.probability < prob_threshold and remove_uncertain_entities:
      continue
    entity.probability = 1.0
    attributes = []
    for attribute in entity.attributes:
      if attribute.value:
        candidate, original_prob = most_likely_candidate(attribute.value)
        attribute.value = [candidate]
      else:
        continue
      if original_prob < prob_threshold and remove_uncertain_attributes:
        continue
      attributes.append(attribute)
    if attributes:
      entity.attributes = attributes
    else:
      continue
    all_entities.append(entity)
  new_state.all_entities = all_entities

  all_relations = []
  for relation in new_state.all_relations:
    if relation.value:
      candidate, original_prob = most_likely_candidate(relation.value)
      relation.value = [candidate]
    else:
      continue
    if original_prob < prob_threshold and remove_uncertain_attributes:
      continue
    all_relations.append(relation)
  new_state.all_relations = all_relations
  return new_state


def sample_from_belief_state(
    agent,
    confidence_threshold=0.75,
    importance_threshold=0.2,
    verbalize_action_with_llm=True,
    num_steps=20,
    verbose=True,
    verbose_action=False,
):
  """Sample a state from the belief state."""
  if verbose:
    print('Original agent prompt: ' + agent.prompt)
  cnt = 0
  while not is_confident(
      agent.state,
      confidence_threshold=confidence_threshold,
      importance_threshold=importance_threshold,
  ):
    action = agent.select_action(verbose=verbose_action)
    if action is None:
      break
    if verbalize_action_with_llm and isinstance(agent, ac.AttributeAgent):
      verbalized_action = agent.verbalize_action(action)
    else:
      verbalized_action = action.__str__()

    if verbose:
      print(f'\nVerbalized action: {verbalized_action}\n')
    if isinstance(agent, ac.BinaryAgent) == 'binary':
      if isinstance(action, ac.AttributeCandidateAction):
        prob = action.attribute_candidate.probability
      else:
        prob = action.relation_candidate.probability
      sampled_answer = np.random.choice(['yes', 'no'], p=[prob, 1 - prob])
    elif isinstance(agent, ac.BeliefAgent):
      sample_answer_prompt = prompts.simulated_random_response_user_prompt(
          user_prompt=agent.prompt,
          agent_question=verbalized_action,
          belief_state=agent.state,
      )
      sampled_answer = agent.llm.generate(sample_answer_prompt)
      answer = re.search(r'<answer>([\s\S]*)</answer>', sampled_answer)
      if answer is None:
        raise ValueError('No answer found in the model output.')
      sampled_answer = answer.group(1).strip()
      if verbose:
        print(f'Sampled answer: {sampled_answer}')
    elif isinstance(agent, ac.AttributeAgent):
      # for implicit entity, we answer yes/no according to Bernoulli(Probability
      # of appearing):
      if action.entity.entity_type == 'implicit':
        entity_exist_prob = action.entity.probability
        sampled_answer = 'yes' if random.random() < entity_exist_prob else 'no'
      else:
        candidate_names = [c.name for c in action.attribute.value]
        candidate_probabilities = [
            c.probability for c in action.attribute.value
        ]
        sampled_answer = np.random.choice(
            candidate_names, p=candidate_probabilities
        )
      if verbose:
        print(f'Sampled answer: {sampled_answer}')
    else:
      raise ValueError('Agent type is not supported.')
    agent.transition(action, sampled_answer, verbose=verbose)
    cnt += 1
    if cnt > num_steps:
      break
  return agent.state


def approximate_state_of_long_prompt(
    prompt: str,
    llm_client: utils.LLM,
    add_implicit_entities: bool = True,
    prob_threshold: float = 0.7,
    verbose: bool = True,
) -> dc.BeliefState:
  """Returns the approximate state of a long prompt.

  Args:
    prompt: a long prompt with a lot of information about the image.
    llm_client: utils.LLM
    add_implicit_entities: whether or not adding implicit entities/background
    prob_threshold: the threshold for the probability of the attribute
      candidate. The attribute candidate with probability less than
      prob_threshold will be ignored.
    verbose: whether or not print intermediate results.

  Returns:
    A belief state with probabilities set to 1 for all
    entities/attributes/relations.
  """
  entity_parser = parser.EntityParser(
      llm_client, add_implicit_entities, verbose=verbose
  )
  relation_parser = parser.RelationParser(llm_client, verbose=verbose)
  attribute_parser = parser.AttributeParser(llm_client, verbose=verbose)
  scene = ac.get_belief_state_from_prompt(
      prompt, entity_parser, relation_parser, attribute_parser
  )
  gt_scene = most_likely_deterministic_state(
      scene,
      remove_uncertain_attributes=True,
      remove_uncertain_entities=True,
      prob_threshold=prob_threshold,
  )
  return gt_scene
