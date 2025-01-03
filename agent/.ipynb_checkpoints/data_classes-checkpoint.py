"""Data classes for the entities and attributes."""

import dataclasses
from typing import Any, List


@dataclasses.dataclass
class Candidate:
  """A candidate is a possible value that an attribute or a relation can take.

  name: name of the candidate
  probability: probability that the user wants this candidate
  """

  name: str
  probability: str | float | None

  def to_json(self):
    return {
        'name': self.name,
        'probability': self.probability,
    }

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]):
    return Candidate(
        name=input_dict['name'],
        probability=input_dict['probability'],
    )


@dataclasses.dataclass
class Attribute:
  """An attribute is a property of an entity.

  It has a name, a value, and an importance score. The importance score
  represents the importance level of the attribute from 0 to 1. in_the_prompt
  represents whether the attribute has appeared in the user prompt. For example,
  For a rabbit: color has a high importance score and teech is a low importance
  score.
  """

  name: str
  value: List[Candidate] | None
  importance_score: str | float | None

  def __str__(self):
    return_string = ''
    return_string += f'Attribute Name: {self.name}'
    if self.importance_score:
      return_string += f', Importance to ask Score: {self.importance_score}'
    if self.value:
      return_string += ', Candidates: ['
      return_string += ', '.join(
          [f'{cand.name}: {round(cand.probability, 2)}' for cand in self.value]
      )
      return_string += ']'
    return return_string.strip()

  def __repr__(self):
    return self.__str__()

  def to_json(self):
    return {
        'name': self.name,
        'value': [x.to_json() for x in self.value],
        'importance_score': self.importance_score,
    }

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]):
    return Attribute(
        name=input_dict['name'],
        value=[Candidate.load_from_json(x) for x in input_dict['value']],
        importance_score=input_dict['importance_score'],
    )


@dataclasses.dataclass
class Relation(Attribute):
  """Relational Attribute describing how two entities are spatially oriented.

  There can be several possibilities such as "part of", "under", "overlap",
  "behind". Describe in free-form text. Relation between two entities can be
  None if nothing is mentioned. `value` is the preposition between the 2
  entities such as 'part of', 'under' or 'overlap', 'behind'. importance score
  is the
  importance score of the relation between the two entities from 0 to 1. For
  example, if the prompt is "rabbit running on grass", the importance of "rabbit
  on grass" is 1.0, and the importance of "rabbit in a park" is 0.2.
  name_entity_1 is the name of the first entity. For example, if the relation is
  "rabbit running on grass", the name of the first entity is "rabbit".
  name_entity_2 is the name of the second entity. For example, if the relation
  is "rabbit running on grass", the name of the second entity is "grass".
  is_bidrectional is True if the relation is bidirectional. For example, if the
  relation is "rabbit is near dog", is_bidirectional is True. If the
  relation is "rabbit running under the tree", is_bidirectional is False.
  """

  description: str
  name_entity_1: str
  name_entity_2: str
  is_bidirectional: bool

  def __str__(self):
    return_string = ''
    if self.name:
      return_string += (
          f'\nName: {self.name}, Entity_1: {self.name_entity_1}, Entity_2:'
          f' {self.name_entity_2}, Importance to ask Score:'
          f' {self.importance_score}, Descriptions: {self.description}'
      )
    if self.value:
      return_string += '\n  Spatial Relation: ['
      return_string += ', '.join(
          [f'{cand.name}: {cand.probability}' for cand in self.value]
      )
      return_string += ']'
    return return_string.strip()

  def __repr__(self):
    return self.__str__()

  def to_json(self):
    return {
        'name': self.name,
        'name_entity_1': self.name_entity_1,
        'name_entity_2': self.name_entity_2,
        'is_bidirectional': self.is_bidirectional,
        'description': self.description,
        'importance_score': self.importance_score,
        'value': [x.to_json() for x in self.value],
    }

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]):
    return Relation(
        name=input_dict['name'],
        name_entity_1=input_dict['name_entity_1'],
        name_entity_2=input_dict['name_entity_2'],
        is_bidirectional=input_dict['is_bidirectional'],
        description=input_dict['description'],
        importance_score=input_dict['importance_score'],
        value=[Candidate.load_from_json(x) for x in input_dict['value']],
    )


@dataclasses.dataclass
class Entity:
  """An entity is a real world object.

  It has the following attributes:

    name: name of the entity.
    importance_score: the importance of the entity from 0 to 1. For example, if
    the prompt is "rabbit running on grass", the importance of "rabbit" is
    1.0, and the importance of "park" is 0.2.
    attributes: A list of important attributes for the entity.
    descriptions: the descriptions for the entity.
    entity_type: The type of the entity. For example, "explicit", "implicit".
    probability: The probability of the entity appearing in the image.
  """

  name: str
  importance_score: str | float | None = None
  attributes: List[Attribute] | None = None
  descriptions: str | None = None
  entity_type: str | None = None
  probability: str | float | None = 1.0

  def __str__(self):
    return_string = ''
    if self.name:
      return_string += f'\nName: {self.name}'
    if self.descriptions:
      return_string += f', Descriptions: {self.descriptions}'
    if self.importance_score:
      return_string += f', Importance to ask Score: {self.importance_score}'
    if self.entity_type:
      return_string += f', Entity Type: {self.entity_type}'
    if self.probability:
      return_string += f', Probability of appearing: {self.probability}'
    return_string += '\n'
    if self.attributes:
      return_string += '  Attributes: \n'
      for att in self.attributes:
        return_string += f'    {att.__str__()}\n'
      return_string += '\n'
    return return_string.strip()

  def __repr__(self):
    return self.__str__()

  def to_json(self):
    return {
        'name': self.name,
        'importance_score': self.importance_score,
        'attributes': [x.to_json() for x in self.attributes],
        'descriptions': self.descriptions,
        'entity_type': self.entity_type,
        'probability': self.probability,
    }

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]):
    return Entity(
        name=input_dict['name'],
        importance_score=input_dict['importance_score'],
        attributes=[
            Attribute.load_from_json(x) for x in input_dict['attributes']
        ],
        descriptions=input_dict['descriptions'],
        entity_type=input_dict['entity_type'],
        probability=input_dict['probability'],
    )


@dataclasses.dataclass
class BeliefState:
  """A scene graph contains information about the scene in terms of the entities and the relations between them.

  all_entities is a list of entities as defined in the Entity Class. Every pair
  of entities in all_entities has a relation. all_relations is a list of
  relation entities. Where each Relation object is the relation between two
  entities.
  """

  all_entities: List[Entity]
  all_relations: List[Relation]
  prompt: str | None = None

  def __str__(self):
    return_string = ''
    return_string += '========== all_entities ==============\n'
    for ent in self.all_entities:
      return_string += f'{ent.__str__()}\n'
    return_string += '========== all_relations ==============\n'
    for rel in self.all_relations:
      return_string += f'{rel.__str__()}\n'
    return return_string.strip()

  def __repr__(self):
    return self.__str__()

  def to_json(self):
    return {
        'all_entities': [x.to_json() for x in self.all_entities],
        'all_relations': [x.to_json() for x in self.all_relations],
        'prompt': self.prompt,
    }

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]):
    return BeliefState(
        all_entities=[
            Entity.load_from_json(x) for x in input_dict['all_entities']
        ],
        all_relations=[
            Relation.load_from_json(x) for x in input_dict['all_relations']
        ],
        prompt=input_dict['prompt'],
    )
