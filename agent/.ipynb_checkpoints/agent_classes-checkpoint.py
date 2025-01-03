"""Classes for agents, their actions, states, etc."""

import abc
import copy
import string
import textwrap
import dataclasses
from typing import Any, List
import numpy as np

from agent import data_classes as dc
from agent import llm_parser as parser
from agent import prompts
from agent import utility_classes as uc


def entropy(probs: list[float]) -> float:
  return -sum(p * np.log(p) for p in probs if p > 0)


def get_belief_state_from_prompt(
    text_prompt: str,
    entity_parser: parser.EntityParser,
    relation_parser: parser.RelationParser,
    attribute_parser: parser.AttributeParser,
) -> dc.BeliefState:
  """Returns the belief state from the prompt.

  Args:
    text_prompt: text prompt to be used to generate the belief state.
    entity_parser: The entity parser used to generate the entities.
    relation_parser: The relation parser used to generate the relations.
    attribute_parser: The attribute parser used to generate the attributes.

  Returns:
    The belief state.
  """
  entities = entity_parser.run(text_prompt)

  # relation and attribute parser can run parallelly:
  def _parse_attributes_or_relation(is_parse_attribute: bool):
    if is_parse_attribute:
      return parser.entity_attributes_generation(
          text_prompt, entities, attribute_parser
      )
    else:
      return relation_parser.run(text_prompt, entities)

  results = [
      _parse_attributes_or_relation(is_parse_attribute)
      for is_parse_attribute in [True, False]
  ]
  return dc.BeliefState(
      all_entities=entities, all_relations=results[1], prompt=text_prompt
  )


def get_belief_state_from_prompt_and_info(
    text_prompt: str,
    entities: List[dc.Entity],
    relations: List[dc.Relation],
    attribute_parser: parser.AttributeParser,
) -> dc.BeliefState:
  """Update the belief state based on the prompt, entity and relations.

  This function only regenerates the attributes of the entities.
  TODO: Regenerate relations based on the new information.


  Args:
    text_prompt: The text prompt to be used to generate the new information.
    entities: The list of entities in the current belief state.
    relations: The list of relations in the current belief state.
    attribute_parser: The attribute parser used to generate the new attributes.

  Returns:
    The new belief state.
  """
  parser.entity_attributes_generation(text_prompt, entities, attribute_parser)
  return dc.BeliefState(
      all_entities=entities, all_relations=relations, prompt=text_prompt
  )


class Action(abc.ABC):
  """Base class for actions. Need to further define the abstract methods."""

  verbalized_action: str | None = None


@dataclasses.dataclass
class AttributeCandidateAction(Action):
  """Action that questions if an attribute candidate has probability 1 or 0.

  For example, if the entity is a "rabbit", the attribute is "color", and the
  attribute candidate is "red", then the question would be "is the color of the
  rabbit red?". We can also write a prompt based on the action and let an LLM
  output the question to be asked by the agent.
  """

  entity: dc.Entity
  attribute: dc.Attribute
  attribute_candidate: dc.Candidate
  verbalized_action: str | None = None

  def __str__(self):
    return (
        f'Is the {self.attribute.name} of'
        f' {self.entity.name} {self.attribute_candidate.name}?'
    )

  def __repr__(self):
    return (
        f'entity name: {self.entity.name}\n'
        f'attribute name: {self.attribute.name}\n'
        f'attribute candidate name: {self.attribute_candidate.name}\n'
    )

  def to_json(self):
    return {
        'entity': self.entity.to_json(),
        'attribute': self.attribute.to_json(),
        'attribute_candidate': self.attribute_candidate.to_json(),
        'verbalized_action': self.verbalized_action,
    }

  @staticmethod
  def load_from_json(input_dict: dict[str, Any]):
    return AttributeCandidateAction(
        entity=dc.Entity.load_from_json(input_dict['entity']),
        attribute=dc.Attribute.load_from_json(input_dict['attribute']),
        attribute_candidate=dc.Candidate.load_from_json(
            input_dict['attribute_candidate']
        ),
        verbalized_action=input_dict['verbalized_action'],
    )


@dataclasses.dataclass
class RelationCandidateAction(Action):
  """Action that questions if an attribute candidate has probability 1 or 0.

  For example, if the entity is a "rabbit", the attribute is "color", and the
  attribute candidate is "red", then the question would be "is the color of the
  rabbit red?". We can also write a prompt based on the action and let an LLM
  output the question to be asked by the agent.
  """

  relation: dc.Relation
  relation_candidate: dc.Candidate
  verbalized_action: str | None = None

  def __str__(self):
    return (
        f'Is {self.relation.name_entity_1} {self.relation_candidate.name}'
        f' {self.relation.name_entity_2}?'
    )

  def __repr__(self):
    return (
        f'relation name: {self.relation.name}\n'
        f'relation description: {self.relation.description}\n'
        f'relation candidate name: {self.relation_candidate.name}\n'
    )

  def to_json(self):
    return {
        'relation': self.relation.to_json(),
        'relation_candidate': self.relation_candidate.to_json(),
        'verbalized_action': self.verbalized_action,
    }

  @staticmethod
  def load_from_json(input_dict: dict[str, Any]):
    return RelationCandidateAction(
        relation=dc.Relation.load_from_json(input_dict['relation']),
        relation_candidate=dc.Candidate.load_from_json(
            input_dict['relation_candidate']
        ),
        verbalized_action=input_dict['verbalized_action'],
    )


@dataclasses.dataclass
class AttributeAction(Action):
  """Action that questions an attribute of an entity.

  For example, if the entity is a "rabbit", the attribute is "color", then the
  question would be "what is the color of the rabbit?". We can also write a
  prompt based on the action and let an LLM output the question to be asked by
  the agent.
  """

  attribute: dc.Attribute
  entity: dc.Entity | None = None  #  If None, the attribute is relation
  verbalized_action: str | None = None

  def verbalize(self):
    """Generate question based on the selected attribute/entity."""
    if self.entity is not None:
      if self.entity.entity_type == 'implicit':
        # For implicit entity, ask whether the entity exist at first:
        self.verbalized_action = (
            f'Is the entity {self.entity.name} in the image?'
        )
      else:
        self.verbalized_action = (
            f'What is the {self.attribute.name} of {self.entity.name}?'
        )
    else:
      self.verbalized_action = (
          f'What is the value of the relation {self.attribute.name}?'
      )

  def __str__(self):
    if self.verbalized_action is None:
      self.verbalize()
    return self.verbalized_action

  def __repr__(self):
    if self.entity is not None:
      return f'entity name: {self.entity.name}\n'
    else:
      return (
          f'attribute name: {self.attribute.name}\n'
          f'verbalized action: {self.verbalized_action}'
      )

  def to_json(self):
    return {
        'attribute': self.attribute.to_json(),
        'entity': self.entity.to_json() if self.entity is not None else None,
        'verbalized_action': self.verbalized_action,
    }

  @staticmethod
  def load_from_json(input_dict: dict[str, Any]):
    return AttributeAction(
        attribute=dc.Attribute.load_from_json(input_dict['attribute']),
        entity=dc.Entity.load_from_json(input_dict['entity'])
        if input_dict['entity'] is not None
        else None,
        verbalized_action=input_dict['verbalized_action'],
    )


def attribute_action_to_prompt(
    action: AttributeAction, template: string.Template | None = None
):
  """Define a prompt to be used for the LLM to generate the question.

  Args:
    action: AttributeAction(entity, attribute).
    template: A string template that has $entity, $attribute and $candidates.

  Returns:
    A string that is the prompt to be used for the LLM to generate the
    question.
  """
  if template is None:
    template = string.Template(textwrap.dedent("""\
      You are constructing a text-to-image (T2I) prompt and want more details from the user.
      You have to ask a question about the the most important entity or the attribute of the most important entity.
      We have entity types: (i) explicit: directly ask question with options; (ii) implicit: ask whether this entity required for the image with yes or no as options; (iii) background: ignore the attribute value and directly ask the value of the entity. (iv) relation: add keyword like `relation` to emphasize this entity is a relation.
      Construct a simple question that directly asks this information from the user and also provides option that the user can pick from. Ask only one question and follow it with options.


      Example1:
      entity: rabbit
      attribute: color
      candidates: black, white, brown
      entity_type: explicit
      question: What color of the rabbit do you have in mind? a. black, b. white, c. brown. d. unkown. If none of these options, what color of the rabbit do you have in mind?

      Example2:
      entity: table
      attribute: size
      candidates: small, medium, large
      entity_type: implicit
      question: Is the entity `table` present in the image?

      Example3:
      entity: weather
      attribute: weather
      candidates: rainy, sunny, cloudy
      entity_type: background
      question: What should the weather be? a. rainy, b. sunny, c. cloudy d. unkown. If none of these options, what should the weather be?

      Example4:
      entity: pig-sky
      attribute: spatial_relation
      candidates: in, below, within
      entity_type: relation
      question: What should the relation pig-sky be? a. above, b. below, c. within d. unkown. If none of these options, what should the relation pig-sky be?

      Example5:
      entity: $entity
      attribute: $attribute
      candidates: $candidates
      entity_type: $entity_type
      question: """))
  cand_dict = {att.name: att.probability for att in action.attribute.value}
  candidates = [
      k for k, _ in sorted(cand_dict.items(), key=lambda item: -item[1])
  ][:5]
  prompt = template.safe_substitute({
      'entity': (
          action.entity.name
          if action.entity is not None
          else action.attribute.name
      ),
      'attribute': action.attribute.name,
      'candidates': ', '.join(candidates),
      'entity_type': (
          action.entity.entity_type if action.entity is not None else 'relation'
      ),
  })
  return prompt


class Agent(abc.ABC):
  """Base class for agents."""

  state: dc.BeliefState
  prompt: str | None = None
  raw_prompt: str | None = None
  history: list[dict[str, Any]]
  json_history: list[dict[str, Any]]

  def __init__(self, state: dc.BeliefState):
    self.state = copy.deepcopy(state)
    self.history = []
    self.json_history = []

  @abc.abstractmethod
  def transition(self, action: Any, observation: Any):
    """Update the state and history based on the action and observation.

    Calling this function will directly change self.state and self.history.

    Args:
      action: The action taken by the agent.
      observation: The observation (i.e. action by the user) observed by the
        agent.
    """
    pass

  @abc.abstractmethod
  def select_action(self) -> Any | None:
    """Select the next action to take based on the current state."""
    pass

  @abc.abstractmethod
  def verbalize_action(self, action: Any) -> str:
    pass

  def save_history(self, action: Any, observation: Any):
    state = copy.deepcopy(self.state)
    action = copy.deepcopy(action)
    self.history.append({
        'state': state,
        'prompt': self.prompt,
        'action': action,
        'observation': observation,
    })
    self.save_history_to_json(action, observation)

  def save_history_to_json(self, action: Any, observation: Any):
    self.json_history.append({
        'state': self.state.to_json() if self.state is not None else None,
        'prompt': self.prompt if self.prompt is not None else None,
        'action': (
            action.to_json()
            if (action is not None and not isinstance(action, str))
            else action
        ),
        'observation': observation,
    })

  def save(self):
    return {
        'state': self.state.to_json(),
        'prompt': self.prompt,
        'raw_prompt': self.raw_prompt,
        'history': self.json_history,
    }

  def load(self, input_dict: dict[str, Any]):
    """Load the agent from a json dictionary.

    Args:
      input_dict: A json dictionary returned by self.save().
    """
    self.state = dc.BeliefState.load_from_json(input_dict['state'])
    self.prompt = input_dict['prompt']
    self.raw_prompt = input_dict['raw_prompt']
    self.json_history = input_dict['history']
    self.history = []
    for item in self.json_history:
      if isinstance(item['action'], dict):
        if 'attribute_candidate' in item['action']:
          action = AttributeCandidateAction.load_from_json(item['action'])
        elif 'relation_candidate' in item['action']:
          action = RelationCandidateAction.load_from_json(item['action'])
        else:
          action = AttributeAction.load_from_json(item['action'])
      else:
        action = item['action']
      self.history.append({
          'state': dc.BeliefState.load_from_json(item['state']),
          'prompt': item['prompt'],
          'action': action,
          'observation': item['observation'],
      })


@dataclasses.dataclass
class AttributeAgentConfig:
  add_implicit_entities: bool = True
  verbose: bool = False
  add_implicit_entity_relation: bool = False
  use_parallel: bool = True
  self_termination: bool = (
      False  # if true, agent will terminate using its own judgement
  )


class AttributeAgent(Agent):
  """Agent that can only ask what an attribute of an entity is.

  For example, the agent can ask "what is the color of the rabbit?".
  """

  def __init__(
      self,
      prompt: str,
      llm: uc.LLM,
      state: dc.BeliefState | None = None,
      config: AttributeAgentConfig = AttributeAgentConfig(),
  ):
    self._config = config
    self.entity_parser = parser.EntityParser(
        llm, self._config.add_implicit_entities, verbose=self._config.verbose
    )
    self.relation_parser = parser.RelationParser(
        llm,
        verbose=self._config.verbose,
        add_implicit_entity_relation=self._config.add_implicit_entity_relation,
    )
    self.attribute_parser = parser.AttributeParser(
        llm, verbose=self._config.verbose
    )
    state = (
        get_belief_state_from_prompt( 
            prompt,
            self.entity_parser,
            self.relation_parser,
            self.attribute_parser,
        )
        if state is None
        else state
    )
    super().__init__(state=state)
    self.llm = llm
    self.prompt = prompt
    self.raw_prompt = prompt

  def verbalize_action(
      self, action: AttributeAction, verbose: bool | None = None
  ) -> str:
    prompt = attribute_action_to_prompt(action)
    verbose = verbose if verbose is not None else self._config.verbose
    if verbose:
      print('Prompt for action verbalization: ' + prompt)
    verbalized_action = self.llm.generate(prompt)
    action.verbalized_action = verbalized_action
    return verbalized_action

  def transition(
      self,
      action: AttributeAction,
      observation: str,
      verbose: bool | None = None,
  ):
    """Transition to the next state. Next prompt should be improved."""
    verbose = verbose if verbose is not None else self._config.verbose
    if verbose:
      print('Transitioning...')
    if action.verbalized_action is None:
      action.verbalized_action = self.verbalize_action(action)
    x = (
        'The chat history is as follows:\n question:'
        f' {action.verbalized_action} and answer: {observation}.\n Turn the'
        ' question and action into a single declarative sentence that'
        ' describes the answer - do not phrase it as a question. Example'
        ' output: the firetruck in the image is red.'
    )
    additional_info = self.llm.generate(x)
    if verbose:
      print('Additional info: ' + additional_info)
    prompt_merged_by_llm = self.llm.generate(
        prompts.merge_info_prompt(self.prompt, additional_info)
    )
    if verbose:
      print('New prompt: ' + prompt_merged_by_llm)
    added_additional_info = self.raw_prompt + '\nnew_info: ' + additional_info
    self.raw_prompt = added_additional_info
    self.prompt = prompt_merged_by_llm
    implicit_entity_to_explicit = False
    if action.entity is not None and action.entity.entity_type == 'implicit':
      is_implicit_entity_exist_prompt = (
          'The chat history is as follows:\n question:'
          f' {action.verbalized_action} and answer: {observation}. Does the'
          ' implicit entity exist? Answer with either `yes` or `no`.'
      )
      is_exist = self.llm.generate(is_implicit_entity_exist_prompt)
      if verbose:
        print([
            is_implicit_entity_exist_prompt,
            '\n======Model Response=====\n',
            is_exist,
        ])
      is_exist = 1 if is_exist.lower().strip().startswith('yes') else 0
      if is_exist:
        # if we found the existance of an implicit entity, then we need to
        # re-parse all the entities and relations:
        implicit_entity_to_explicit = True
        self.state = get_belief_state_from_prompt(
            prompt_merged_by_llm,
            self.entity_parser,
            self.relation_parser,
            self.attribute_parser,
        )
    if not implicit_entity_to_explicit:
      self.state = get_belief_state_from_prompt_and_info(
          prompt_merged_by_llm,
          self.state.all_entities,
          self.state.all_relations,
          self.attribute_parser,
      )

    # Post-process state to add forgotten attributes and relations.
    for history_item in self.history:
      history_action = history_item['action']
      history_observation = history_item['observation']
      assert isinstance(
          history_action, AttributeAction
      ), 'history_action must be an AttributeAction.'
      entity_name = None
      if history_action.entity is not None:
        entity_name = history_action.entity.name
        if history_action.entity.entity_type == 'implicit':
          continue
      attribute_name = history_action.attribute.name

      if entity_name is None:
        # relation action
        relation_found = None
        for relation in self.state.all_relations:
          if relation.name == attribute_name:
            relation_found = relation
            break
        relation = relation_found
        if relation_found is None:
          relation = copy.deepcopy(history_action.attribute)
          self.state.all_relations.append(relation)
        relation.importance_score = 0
        relation.value = [
            dc.Candidate(name=history_observation, probability=1.0)
        ]
      else:
        # attribute action
        entity_found = None
        for entity in self.state.all_entities:
          if entity.name == entity_name:
            entity_found = entity
            break
        if entity_found is not None:
          # Entity found, set the corresponding attribute importance score to 0.
          attribute_found = None
          for attribute in entity_found.attributes:
            if attribute.name == attribute_name:
              attribute_found = attribute
              break
          attribute = attribute_found
          if attribute_found is None:
            attribute = copy.deepcopy(history_action.attribute)
            if entity_found.attributes:
              entity_found.attributes.append(attribute)
            else:
              entity_found.attributes = [attribute]
          attribute.importance_score = 0
          attribute.value = [
              dc.Candidate(name=history_observation, probability=1.0)
          ]
        else:
          # Entity not found, add it to the state.
          attribute = copy.deepcopy(history_action.attribute)
          attribute.importance_score = 0
          attribute.value = [
              dc.Candidate(name=history_observation, probability=1.0)
          ]
          self.state.all_entities.append(
              dc.Entity(
                  name=entity_name,
                  importance_score=1.0,
                  attributes=[attribute],
                  entity_type='asked but forgotten',
              )
          )
    self.save_history(action, observation)

  def select_action(
      self, verbose: bool | None = None, minmax_score_thre=0.7
  ) -> AttributeAction | None:
    """Return important (entity, attribute) pair with highest importance score.

    Here relation is not being used. Needs to be improved.

    Args:
      verbose: If true, print the entity, attribute, and score.
      minmax_score_thre: the threshold of minmax_score to stop taking action.
        minmax_score is the minimal score of the highest candidate probability.
    """
    verbose = verbose if verbose is not None else self._config.verbose
    minmax_score = 1.0
    if not self._config.self_termination:
      minmax_score_thre = np.inf
    if verbose:
      print('Selecting action...')
    (
        highest_attribute_score,
        most_important_entity,
        most_important_attribute,
    ) = (float('-inf'), None, None)
    for entity in self.state.all_entities:
      for attribute in entity.attributes:
        minmax_score = min(
            minmax_score,
            max([x.probability for x in attribute.value]),
        )
        # we do not consider attribute with high candidate probability:
        if max([x.probability for x in attribute.value]) > minmax_score_thre:
          continue
        cand_entropy = entropy([x.probability for x in attribute.value])
        overall_score = (
            float(entity.importance_score)
            * float(attribute.importance_score)
            * float(entity.probability)
            * cand_entropy
        )
        if overall_score > highest_attribute_score:
          highest_attribute_score = overall_score
          most_important_entity = entity
          most_important_attribute = attribute
          if verbose:
            print(
                f'Entity: {entity.name}, Attribute: {attribute.name}, Score:'
                f' {overall_score}'
            )

    # Check overall score for relations:
    highest_relation_score, most_important_relation = (
        float('-inf'),
        None,
    )
    for relation in self.state.all_relations:
      if not relation.value:
        continue
      minmax_score = min(
          minmax_score, max([x.probability for x in relation.value])
      )
      cand_entropy = entropy([x.probability for x in relation.value])
      overall_score = float(relation.importance_score) * cand_entropy
      if overall_score > highest_relation_score:
        highest_relation_score = overall_score
        most_important_relation = relation
        if verbose:
          print(
              f'Relation: {relation.name}, entieis: {relation.name_entity_1},'
              f' {relation.name_entity_2}, score: {overall_score}'
          )

    # if the agent is fairly confident for all attributes, we do nothing.
    if minmax_score >= minmax_score_thre:
      return None
    if highest_relation_score < highest_attribute_score:
      action = AttributeAction(
          entity=most_important_entity,
          attribute=most_important_attribute,
      )
      action.verbalize()
    else:
      action = AttributeAction(
          attribute=most_important_relation,
      )
      action.verbalize()

    action.verbalized_action = action.__str__()
    if verbose:
      print(f'Selected action: {action}')
    return action


class BeliefAgent(Agent):
  """Agent that uses LLM to select actions."""

  def __init__(
      self,
      prompt: str,
      llm: uc.LLM,
      state: dc.BeliefState | None = None,
      config: AttributeAgentConfig = AttributeAgentConfig(),
  ):
    self._config = config
    self.entity_parser = parser.EntityParser(
        llm, self._config.add_implicit_entities, verbose=self._config.verbose
    )
    self.relation_parser = parser.RelationParser(
        llm,
        verbose=self._config.verbose,
        add_implicit_entity_relation=self._config.add_implicit_entity_relation,
    )
    self.attribute_parser = parser.AttributeParser(
        llm, verbose=self._config.verbose
    )
    state = (
        get_belief_state_from_prompt(
            prompt,
            self.entity_parser,
            self.relation_parser,
            self.attribute_parser,
        )
        if state is None
        else state
    )
    super().__init__(state=state)
    self.llm = llm
    self.prompt = prompt
    self.original_prompt = prompt
    self.action_parser = parser.ActionParser(llm, verbose=config.verbose)
    self.answer_parser = parser.AnswerParser(llm, verbose=config.verbose)
    self.raw_prompt = prompt

  def select_action(
      self, use_original_prompt=False, verbose: bool = False
  ) -> str:
    """Select an action using LLM."""
    if verbose:
      print('Selecting action...')
    self.action_parser.verbose = verbose
    if use_original_prompt:
      prompt = self.original_prompt
    else:
      prompt = self.prompt
    action = self.action_parser.run(prompt, self.state, self.history)
    return action

  def verbalize_action(self, action: str) -> str:
    return action

  def transition(
      self, action: str, observation: str, verbose: bool | None = None
  ):
    x = (
        'The chat history is as follows:\n question:'
        f' {action} and answer: {observation}.\n Turn the'
        ' question and action into a single declarative sentence that'
        ' describes the answer - do not phrase it as a question. Example'
        ' output: the firetruck in the image is red.'
    )
    additional_info = self.llm.generate(x)
    if verbose:
      print('Additional info: ' + additional_info)
    prompt_merged_by_llm = self.llm.generate(
        prompts.merge_info_prompt(self.prompt, additional_info)
    )
    if verbose:
      print('New prompt: ' + prompt_merged_by_llm)
    added_additional_info = self.raw_prompt + '\nnew_info: ' + additional_info
    self.raw_prompt = added_additional_info
    self.prompt = prompt_merged_by_llm
    self.state = get_belief_state_from_prompt(
        self.prompt,
        self.entity_parser,
        self.relation_parser,
        self.attribute_parser,
    )
    self.save_history(action, observation)

  def answer_question(self, question: str, verbose: bool = False) -> str:
    """Answer the question using LLM."""
    if verbose:
      print(f'Answering question: {question}')
    self.answer_parser.verbose = verbose
    answer = self.answer_parser.run(self.prompt, question, self.state)
    return answer


class BinaryAgent(Agent):
  """Agent that can only ask yes/no questions about attribute candidates.

  For example, the agent can ask "is the color of the rabbit red?".
  """

  def parse_observation(self, observation: str | bool) -> bool:
    if isinstance(observation, str):
      is_candidate_selected = (
          True
          if observation.lower() == 'yes' or observation.lower() == 'y'
          else False
      )
    else:
      is_candidate_selected = observation
    return is_candidate_selected

  def verbalize_action(self, action: AttributeCandidateAction) -> str:
    verbalized_action = (
        f'Is the {action.attribute.name} of'
        f' {action.entity.name} {action.attribute_candidate.name}?'
        'If it is not any of these what is it?'
    )
    action.verbalized_action = verbalized_action
    return verbalized_action

  def transition(
      self,
      action: AttributeCandidateAction | RelationCandidateAction,
      observation: str | bool,
  ):
    """Set the probability of the attribute candidate to 1 or 0 based on the observation.

    The action includes entity, attribute, and attribute candidate. The action
    is parsed to a question to be asked by the agent (whether the attribute of
    the entity is the attribute candidate). The observation is the user response
    observed by the agent.
    If the user says yes, the probability of the attribute candidate is set to
    1. All other candidates are removed.
    If the user says no, the attribute candidate probability is set to 0, and
    the probability of the other candidates are updated.

    Args:
      action: AttributeCandidateAction(entity, attribute, attribute_candidate).
      observation: The user response to the question asked by the agent, either
        yes or no.

    Returns:
      The next state of the agent.
    """
    is_candidate_selected = self.parse_observation(observation)
    if isinstance(action, AttributeCandidateAction):
      entity = action.entity
      attribute = action.attribute
      candidate = action.attribute_candidate
      assert (
          attribute in entity.attributes
      ), 'action.attribute must be in action.entity.attributes.'
      assert (
          candidate in attribute.value
      ), 'action.attribute_candidate must be in action.attribute.value.'
    else:
      attribute = action.relation
      candidate = action.relation_candidate

    original_probability = candidate.probability
    if is_candidate_selected:
      candidate.probability = 1.0
      attribute.value = [candidate]
    else:
      candidate.probability = 0.0
      value = []
      for c in attribute.value:
        if c.name != candidate.name:
          if original_probability == 1.0:
            c.probability = 1 / (len(attribute.value) - 1)
          else:
            c.probability = c.probability / (1 - original_probability)
        value.append(c)
      attribute.value = value
    self.save_history(action, observation)
    return self.state

  def action_heuristic(
      self, action: AttributeCandidateAction | RelationCandidateAction
  ):
    """Heuristic scores for the action.

    Args:
      action: AttributeCandidateAction(entity, attribute, attribute_candidate)
        or RelationCandidateAction(relation, relation_candidate).

    Returns:
      heuristic score of the action, evaluated as:
      entity.importance_score * attribute.importance_score *
      attribute_candidate.probability * (1 - attribute_candidate.probability).
        This ensures that the heuristic is high if the probability is close to
        0.5 (the agent is unsure).

    Raises:
      ValueError: Entity importance score must be a float.
      ValueError: Attribute importance score must be a float.
      ValueError: Attribute candidate probability must be a float.
      ValueError: Attribute candidate probability must be between 0 and 1.
    """
    if isinstance(action, AttributeCandidateAction):
      entity = action.entity
      attribute = action.attribute
      attribute_candidate = action.attribute_candidate
      probability = attribute_candidate.probability
      if not isinstance(entity.importance_score, float):
        raise ValueError('Entity importance score must be a float.')
      if not isinstance(attribute.importance_score, float):
        raise ValueError('Attribute importance score must be a float.')
      if not isinstance(attribute_candidate.probability, float):
        raise ValueError('Attribute candidate probability must be a float.')
      if probability < 0.0 or probability > 1.01:
        raise ValueError(
            'Attribute candidate probability must be between 0 and 1.'
        )
      return (
          entity.importance_score
          * attribute.importance_score
          * probability
          * (1 - probability)
      )
    else:
      relation = action.relation
      relation_candidate = action.relation_candidate
      probability = relation_candidate.probability
      if not isinstance(relation.importance_score, float):
        raise ValueError('Relation importance score must be a float.')
      if not isinstance(relation_candidate.probability, float):
        raise ValueError('Relation candidate probability must be a float.')
      if probability < 0.0 or probability > 1.01:
        raise ValueError(
            'Relation candidate probability must be between 0 and 1.'
        )
      return (
          relation.importance_score
          * 0.5
          * relation_candidate.probability
          * (1 - relation_candidate.probability)
      )

  def select_action(
      self, verbose: bool = False
  ) -> AttributeCandidateAction | RelationCandidateAction | None:
    best_action = None
    best_action_heuristic = -1.0
    first_action = True
    for e in self.state.all_entities:
      for a in e.attributes:
        for ac in a.value:
          action = AttributeCandidateAction(
              entity=e, attribute=a, attribute_candidate=ac
          )
          heuristic = self.action_heuristic(action)
          if verbose:
            print(
                f'Action: {action}, Heuristic: {heuristic}, Probability:'
                f' {ac.probability}'
            )
          if heuristic > best_action_heuristic or first_action:
            best_action_heuristic = heuristic
            best_action = action
            first_action = False
    for r in self.state.all_relations:
      for rc in r.value:
        action = RelationCandidateAction(relation=r, relation_candidate=rc)
        heuristic = self.action_heuristic(action)
        if verbose:
          print(
              f'Action: {action}, Heuristic: {heuristic}, Probability:'
              f' {rc.probability}'
          )
        if heuristic > best_action_heuristic:
          best_action_heuristic = heuristic
          best_action = action
    return best_action


class LLMAgent(Agent):
  """Agent that uses LLM to select actions."""

  def __init__(
      self,
      prompt: str,
      llm: uc.LLM,
      state: dc.BeliefState | None = None,
      config: AttributeAgentConfig = AttributeAgentConfig(),
  ):
    self._config = config
    self.entity_parser = parser.EntityParser(
        llm, self._config.add_implicit_entities, verbose=self._config.verbose
    )
    self.relation_parser = parser.RelationParser(
        llm,
        verbose=self._config.verbose,
        add_implicit_entity_relation=self._config.add_implicit_entity_relation,
    )
    self.attribute_parser = parser.AttributeParser(
        llm, verbose=self._config.verbose
    )
    state = (
        get_belief_state_from_prompt( 
            prompt,
            self.entity_parser,
            self.relation_parser,
            self.attribute_parser,
        )
        if state is None
        else state
    )
    super().__init__(state=state)
    self.llm = llm
    self.prompt = prompt
    self.original_prompt = prompt
    self.action_parser = parser.ActionParser(llm, verbose=config.verbose)
    self.answer_parser = parser.AnswerParser(llm, verbose=config.verbose)
    self.raw_prompt = prompt
    self.chat_history = ''

  def create_next_question_prompt(self):
    return (
        'Based on the chat history please provide a new question to ask about'
        ' the image. the chat history is as follows and is enclosed in'
        ' <chat_history> and </chat_history>'
        f' markers:{self.chat_history} </chat_history> The question should be'
        ' as concise and direct as possible. The question should aim to learn'
        ' more about the attributes and contents of the image, the objects,'
        ' the spatial layout, and the style. Make sure that you question the'
        ' answer within <question> and </question> markers.'
    )

  def select_action(self) -> str:
    """Select an action using LLM."""
    first_question = (
        f'The orginal prompt was: {self.original_prompt} - Based on the'
        ' original prompt please provide a question to ask about the image.'
        ' The question should be as concise and direct as possible. The'
        ' question should aim to learn more about the attributes and contents'
        ' of the image, the objects, the spatial layout, and the style. Make'
        ' sure that you question the answer within <question> and </question>'
        ' markers'
    )
    if self.prompt == self.original_prompt:
      action = self.action_parser.run_generate(first_question)
    else:
      action = self.action_parser.run_generate(
          self.create_next_question_prompt()
      )
    self.chat_history += f'<question> {action} </question>\n'
    return action

  def verbalize_action(self, action: str) -> str:
    return action

  def transition(
      self, action: str, observation: str, verbose: bool | None = None
  ):
    self.chat_history += f'<answer> {observation} </answer>\n'
    x = (
        'The chat history is as follows:\n question:'
        f' {action} and answer: {observation}.\n Turn the'
        ' question and action into a single declarative sentence that'
        ' describes the answer - do not phrase it as a question. Example'
        ' output: the firetruck in the image is red.'
    )
    additional_info = self.llm.generate(x)
    if verbose:
      print('Additional info: ' + additional_info)
    prompt_merged_by_llm = self.llm.generate(
        prompts.merge_info_prompt(self.prompt, additional_info)
    )
    if verbose:
      print('New prompt: ' + prompt_merged_by_llm)
    added_additional_info = self.raw_prompt + '\nnew_info: ' + additional_info
    self.raw_prompt = added_additional_info
    self.prompt = prompt_merged_by_llm
    self.state = get_belief_state_from_prompt(
        self.prompt,
        self.entity_parser,
        self.relation_parser,
        self.attribute_parser,
    )
    self.save_history(action, observation)

  def answer_question(self, question: str, verbose: bool = True) -> str:
    """Answer the question using LLM."""
    if verbose:
      print(f'Answering question: {question}')
    self.answer_parser.verbose = verbose
    answer = self.answer_parser.run(self.prompt, question, self.state)
    return answer
