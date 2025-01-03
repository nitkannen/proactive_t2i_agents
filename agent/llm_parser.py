"""Parse LLM outputs into data classes."""

import json
import re
from typing import Any, Dict, List, TypeVar
import tenacity

from agent import data_classes
from agent import prompts
from agent import utility_classes as utils

Attribute = data_classes.Attribute
BeliefState = data_classes.BeliefState
Candidate = data_classes.Candidate
Entity = data_classes.Entity
Relation = data_classes.Relation


T = TypeVar('T')
N_RETRIES = 1
IMPORTANCE_SCORE = 'importance_to_ask_score'
PROBABILITY_OF_APPEARING = 'probability_of_appearing'


def _parse_llm_output_to_json(model_output: str) -> List[Dict[str, Any]]:
  """Parses the model output into a json object."""
  if '```' in model_output:
    model_output = model_output.split('```')[1].strip()
  if model_output[:4].lower() == 'json':
    model_output = model_output[4:].strip()
  if ']' in model_output:
    model_output = model_output.split(']')[0] + ']'
  try:
    data = json.loads(model_output)
    return data
  except ValueError as exc:
    print(model_output)
    raise ValueError('Error in parsing json.') from exc


class ActionParser:
  """Parses the action from the model output."""

  def __init__(self, llm: utils.LLM, verbose: bool = False):
    self._llm = llm
    self.verbose = verbose

  def _construct_prompt(
      self,
      user_prompt: str,
      belief_state: BeliefState,
      history: list[dict[str, Any]],
  ) -> str:
    """Constructs the prompt for the action parser.

    Args:
      user_prompt: The user prompt.
      belief_state: The belief state of the agent.
      history: The history of the agent.

    Returns:
      The prompt for the action parser.
    """
    prompt = prompts.select_action_based_on_belief_prompt(
        user_prompt, belief_state, history
    )
    return prompt

  def _extract_question_from_llm_output(self, model_output: str) -> str:
    question = re.search(r'<question>([\s\S]*)</question>', model_output)
    if question is None:
      raise ValueError('No question found in the model output.')
    return question.group(1).strip()

  @tenacity.retry(stop=tenacity.stop_after_attempt(N_RETRIES))
  def run(
      self,
      user_prompt: str,
      belief_state: BeliefState | None,
      history: list[dict[str, Any]],
  ) -> str:
    """Get a list of attributes based on the prompt and entity."""
    prompt = self._construct_prompt(
        user_prompt,
        belief_state=belief_state,
        history=history,
    )
    if self.verbose:
      print('========== action parser prompt ==============\n')
      print(prompt)
    response = self._llm.generate(prompt)
    if self.verbose:
      print('========== action parser response ==============\n')
      print(response)
      print('\n')
    return self._extract_question_from_llm_output(response)

  @tenacity.retry(stop=tenacity.stop_after_attempt(N_RETRIES))
  def run_generate(
      self,
      user_prompt: str,
  ) -> str:
    """Get a list of attributes based on the prompt and entity."""
    if self.verbose:
      print('========== action parser prompt ==============\n')
      print(user_prompt)
    response = self._llm.generate(user_prompt)
    if self.verbose:
      print('========== action parser response ==============\n')
      print(response)
      print('\n')
    return self._extract_question_from_llm_output(response)


class AnswerParser:
  """Parses the action from the model output."""

  def __init__(self, llm: utils.LLM, verbose: bool = False):
    self._llm = llm
    self.verbose = verbose

  def _construct_prompt(
      self, user_prompt: str, agent_question: str, belief_state: BeliefState
  ) -> str:
    """Constructs the prompt for the action parser.

    Args:
      user_prompt: The user prompt.
      agent_question: The question asked by the agent.
      belief_state: The belief state of the user agent.

    Returns:
      The prompt for the action parser.
    """
    prompt = prompts.simulated_belief_user_prompt(
        user_prompt, agent_question, belief_state
    )
    return prompt

  def _extract_answer_from_llm_output(self, model_output: str) -> str:
    answer = re.search(r'<answer>([\s\S]*)</answer>', model_output)
    if answer is None:
      raise ValueError('No answer found in the model output.')
    return answer.group(1).strip()

  @tenacity.retry(stop=tenacity.stop_after_attempt(N_RETRIES))
  def run(
      self,
      user_prompt: str,
      agent_question: str,
      belief_state: BeliefState | None,
  ) -> str:
    """Get a list of attributes based on the prompt and entity."""
    prompt = self._construct_prompt(
        user_prompt,
        agent_question=agent_question,
        belief_state=belief_state,
    )
    if self.verbose:
      print('========== user answer parser prompt ==============\n')
      print(prompt)
    response = self._llm.generate(prompt)
    if self.verbose:
      print('========== user answer parser response ==============\n')
      print(response)
      print('\n')
    return self._extract_answer_from_llm_output(response)


class RelationParser:
  """Parses the relations between entities."""

  def __init__(
      self,
      llm: utils.LLM | None = None,
      verbose: bool = False,
      add_implicit_entity_relation: bool = False,
  ):
    self._llm = (
        utils.LLM( 
            model_id=utils.DEFAULT_LLM_VERTEX_ID, use_json_constraints=True
        )
        if llm is None
        else llm
    )
    self.verbose = verbose
    self._add_implicit_entity_relation = add_implicit_entity_relation

  def _construct_prompt(self, user_prompt: str, entities: List[Entity]) -> str:
    entity_names = []
    for entity in entities:
      if entity is not None and entity.probability != 0:
        if (
            entity.entity_type != 'implicit'
            or self._add_implicit_entity_relation
        ):
          entity_names.append(entity.name)
    return prompts.relation_parser_prompt(user_prompt, entity_names)

  def _construct_relations(self, model_output: str) -> List[Relation]:
    """Parses the model output into a list of attributes."""
    data = _parse_llm_output_to_json(model_output)
    res = []
    for att in data:
      res.append(
          Relation(
              name=att['name'],
              description=att['description'],
              value=[
                  Candidate(name=k, probability=v)
                  for k, v in att['spatial_relation'].items()
              ],
              importance_score=float(att[IMPORTANCE_SCORE]),
              name_entity_1=att['name_entity_1'],
              name_entity_2=att['name_entity_2'],
              is_bidirectional=att['is_bidirectional'],
          )
      )
    return res

  @tenacity.retry(stop=tenacity.stop_after_attempt(N_RETRIES))
  def run(self, user_prompt: str, entities: List[Entity]) -> List[Relation]:
    """Get a list of attributes based on the prompt and entity."""
    prompt = self._construct_prompt(user_prompt, entities)
    response = self._llm.generate(prompt)
    if self.verbose:
      print(response)
    return self._construct_relations(response)


class EntityParser:
  """Parses the entities from the model output."""

  def __init__(
      self,
      llm: utils.LLM,
      add_implicit_entities: bool = False,
      verbose: bool = False,
  ):
    """Init.

    Args:
      llm: The LLM to use for parsing.
      add_implicit_entities: Whether to add implicit entities and background
        entities.
      verbose: whether or not print intermediate results.
    """
    self._llm = llm
    self._add_implicit_entities = add_implicit_entities
    self.verbose = verbose

  def _construct_prompt(self, user_prompt: str) -> str:
    if not self._add_implicit_entities:
      return prompts.entity_parser_prompt(user_prompt)
    else:
      return prompts.entity_parser_prompt_w_background(user_prompt)

  def _construct_entities(self, model_output: str) -> List[Entity]:
    """Parses the model output into a list of attributes."""
    data = _parse_llm_output_to_json(model_output)
    res = []
    for entity in data:
      entity_object = Entity(
          name=entity['name'],
          importance_score=entity[IMPORTANCE_SCORE],
          descriptions=entity['description'],
          probability=entity[PROBABILITY_OF_APPEARING],
      )
      if 'entity_type' in entity:
        entity_object.entity_type = entity['entity_type']
      else:
        entity_object.entity_type = None
      res.append(entity_object)
    return res

  @tenacity.retry(stop=tenacity.stop_after_attempt(N_RETRIES))
  def run(self, user_prompt: str) -> List[Entity]:
    """Get a list of entities based on the prompt."""
    prompt = self._construct_prompt(user_prompt)
    response = self._llm.generate(prompt)
    if self.verbose:
      print(prompt)
      print(response)
    return self._construct_entities(response)


class AttributeParser:
  """Parses the attributes from the model output."""

  def __init__(self, llm: utils.LLM, verbose: bool = False):
    self._llm = llm
    self.verbose = verbose

  def _construct_prompt(
      self, user_prompt: str, entity: Entity, entity_list: List[Entity] | None
  ) -> str:
    if entity.entity_type == 'background':
      return prompts.attribute_parser_prompt_w_background(
          user_prompt, entity, entity_list
      )
    else:
      return prompts.attribute_parser_prompt(user_prompt, entity, entity_list)

  def _construct_attributes(self, model_output: str) -> List[Attribute]:
    """Parses the model output into a list of attributes."""
    data = _parse_llm_output_to_json(model_output)
    res = []
    for att in data:
      candidate_list = att['candidates']
      sum_prob = sum(candidate_list.values())
      if sum_prob <= 0:
        raise ValueError(
            'Sum probabilities of attribute candidates is less than 0.'
        )
      candidate_list = [
          Candidate(name=name, probability=probability / sum_prob)
          for name, probability in candidate_list.items()
      ]
      res.append(
          Attribute(
              name=att['name'],
              importance_score=float(att[IMPORTANCE_SCORE]),
              value=candidate_list,
          )
      )
    return res

  @tenacity.retry(stop=tenacity.stop_after_attempt(N_RETRIES))
  def run(
      self, user_prompt: str, entity: Entity, entity_list: List[Entity] | None
  ) -> List[Attribute]:
    """Get a list of attributes based on the prompt and entity."""
    prompt = self._construct_prompt(user_prompt, entity, entity_list)
    response = self._llm.generate(prompt)
    if self.verbose:
      print(entity)
      print(response)
    return self._construct_attributes(response)


def entity_attributes_generation(
    user_prompt: str,
    entity_list: List[Entity],
    parser: AttributeParser,
    use_parallel: bool = False,
) -> List[Attribute]:
  """Generate a list of attributes for each of the entity in the list."""

  def _one_entity_attributes_gen(
      entity: Entity, entity_list: List[Entity] | None
  ):
    if entity.probability == 0:
      entity.attributes = []
    else:
      entity.attributes = parser.run(user_prompt, entity, entity_list)

  if use_parallel:
    raise ValueError('Parallel processing is not supported in third party.')
  else:
    results = [
        _one_entity_attributes_gen(entity, entity_list)
        for entity in entity_list
    ]
  return results
