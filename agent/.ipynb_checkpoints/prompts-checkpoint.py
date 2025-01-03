"""Prompts for the parsers."""

import textwrap
from typing import Any, List
from agent import data_classes

BeliefState = data_classes.BeliefState
Entity = data_classes.Entity
Attribute = data_classes.Attribute


def merge_info_prompt(prompt: str, additional_info: str) -> str:
  return (
      'You are writing a prompt for a text-to-image model based on user'
      f' feedback. The original prompt is {prompt}. The user has provided some'
      f' additional information: {additional_info}. Please write a new prompt'
      ' for the text-to-image model. The new prompt should be a meaningful'
      ' sentence or a paragraph that combines the original prompt and the'
      ' additional information. Do not add any new information that is not'
      ' mentioned in the prompt or the additional information. Make sure the'
      ' information in the original prompt is not changed. Make sure the'
      ' additional information is included in the new prompt. Make sure the'
      ' new prompt is a description of an image. If the additional information'
      ' or the original prompt specifically says that a thing does not exist'
      ' in the image, you should make sure the new prompt mentions that'
      ' this thing does not exist in the image. DO NOT generate rationale or'
      ' anything that is not part of a description of the image.'
  )


def entity_parser_prompt(user_prompt: str) -> str:
  return textwrap.dedent(f"""\
        Given a text-to-image prompt list out all the entities that are mentioned in the prompt.
        The output should be list and each entry should be formated as a JSON dict with the following fields:

        "name": The name of the entity.
        "importance_to_ask_score": The importance score of asking a question about this entity. Make sure that this is a number between 0 and 1, higher means more important. If the entity is not important in the context of the user prompt or the entity has been fully described in the user prompt, assign a lower score. If the entity is important in the context of the user prompt and there is not enough information in the user prompt to fully describe the entity, assign a higher score.
        "description": A short description of the entity.
        "probability_of_appearing": The probability of the entity appearing in the image. This is a number between 0 and 1. You should assign a probability with the following rules in mind:
        1. If the prompt says an entity does not exist, assign a 0.0 probability. Because the entity does not exist, you should also assign 0 to importance_to_ask_score of this entity.
        2. If the prompt indicates an entity definitly exists in the image, assign a 1.0 probability.
        3. If the prompt says an entity exists but there is an indication that the entity is not likely to appear in the image, assign a probability between 0 and 1, higher if the entity is more likely to appear in the image.

        Below is an example input and output pair:
        Example1:
        Input: {{
          "user_prompt": "generate an image of a lionhead rabbit running on grass with sun shining. There is no trees in the background."
        }}
        Output: [
            {{
              "name": "rabbit",
              "importance_to_ask_score": 0.6,
              "description": "a lionhead rabbit",
              "probability_of_appearing": 1.0
            }},
            {{
              "name": "grass",
              "importance_to_ask_score": 0.5,
              "description": "grass",
              "probability_of_appearing": 1.0
            }},
            {{
              "name": "sun",
              "importance_to_ask_score": 0.1,
              "description": "sun is shining",
              "probability_of_appearing": 0.3
            }},
            {{
              "name": "sun light",
              "importance_to_ask_score": 0.1,
              "description": "sun light shining on the grass and the rabbit",
              "probability_of_appearing": 1.0
            }},
            {{
              "name": "tree",
              "importance_to_ask_score": 0.,
              "description": "trees in the background",
              "probability_of_appearing": 0.
            }},
        ]

        Identify the entities given the input given below. Strictly stick to the format.
        Input: {{
          "user_prompt": "{user_prompt}"
        }}
        Output:""")


def entity_parser_prompt_w_background(user_prompt: str) -> str:
  return textwrap.dedent(f"""\
  Given a text-to-image prompt list out all the entities that are mentioned in the prompt.

  **Explicit Entities:** List all clearly stated entities within the prompt (people, objects, animals, locations, etc.).
  **Implicit Entities:** Identify potential entities that are implied or strongly suggested by the prompt, even if not explicitly mentioned.
  **Background Entities:** Deduce relevant background elements which could impact the image generation from the prompt or context, including:
      **Weather:** If the scene or mood suggests specific weather conditions (sunny, rainy, stormy, etc.).
      **Location:** If a general or specific setting is hinted at (indoors, outdoors, a particular city/landscape, etc.).
      **Time of Day:** If the prompt implies a certain time (dawn, midday, dusk, night).
      **Mood or Atmosphere:** If the prompt evokes a particular emotion or ambiance (joyful, mysterious, peaceful, etc.).


  The output should be list and each entry should be formated as a JSON dict with the following fields:

  "name": The name of the entity.
  "importance_to_ask_score": The importance score of asking a question about this entity to reduce the uncertainty of what the image is given the user prompt. Make sure that this is a number between 0 and 1, higher means more important. Consider these factors when assigning scores: 1. Increate the score for entities that are the primary focus or subject of the prompt; 2. increase the score for entities that could strongly influence the layout of the image, such as the position or portrayal of other entities in the scene; 3. significantlydescrease the score for entities that are already well specified in the prompt; 4. significantlyincrease the score for implicit entities that are likely to appear in the image and their appearance can significantly impact the image.
  "description": A short description of the entity.
  "entity_type": The type of this entitiy. It could be either explicit, implicit, background. No other value is allowed.
  "probability_of_appearing": The probability of the entity appearing in the image. This is a number between 0 and 1. You should assign a probability with the following rules in mind:
    1. If the prompt says an entity does not exist, assign a 0.0 probability. Because the entity does not exist, you should also assign 0 to importance_to_ask_score of this entity.
    2. If the prompt indicates an entity definitly exists in the image, assign a 1.0 probability.
    3. If the prompt does not say anything about the existence of the entity, assign a probability between 0 and 1. This probability is higher if the entity is more likely to appear in the image given the context specified by the prompt.
    4. If the prompt says an entity exists but there is an indication that the entity is not likely to appear in the image, assign a probability between 0 and 1, higher if the entity is more likely to appear in the image.

  Below is an example input and output pair:
  Example1:
  Input: {{
    "user_prompt": "generate an image of a lionhead rabbit running on grass with sun shining. There is no trees in the background."
  }}
  Output: [
      {{
        "name": "rabbit",
        "importance_to_ask_score": 0.5,
        "description": "a lionhead rabbit",
        "entity_type": "explicit",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "grass",
        "importance_to_ask_score": 0.5,
        "description": "grass",
        "entity_type": "explicit",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "sun",
        "importance_to_ask_score": 0.1,
        "description": "sun is shining",
        "entity_type": "explicit",
        "probability_of_appearing": 0.3
      }},
      {{
        "name": "sun light",
        "importance_to_ask_score": 0.1,
        "description": "sun light shining on the grass and the rabbit",
        "entity_type": "explicit",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "tree",
        "importance_to_ask_score": 0,
        "description": "trees in the background",
        "entity_type": "explicit",
        "probability_of_appearing": 0
      }}
      {{
        "name": "camera angle",
        "importance_to_ask_score": 0.8,
        "description": "the camera angle of the image",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "weather",
        "importance_to_ask_score": 0.8,
        "description": "weather",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "image style",
        "importance_to_ask_score": 1.0,
        "description": "the style of the image",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "background color",
        "importance_to_ask_score": 0.8,
        "description": "the background color of the image",
        "entity_type": "background",
        "probability_of_appearing": 0.5
      }}
  ]

  Example2:
  Input: {{
    "user_prompt": "An eagle soars while a dog walks below. There is not a cat in the image."
  }}
  Output: [
      {{
        "name": "eagle",
        "importance_to_ask_score": 0.9,
        "description": "an eagle soaring through the sky",
        "entity_type": "explicit",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "dog",
        "importance_to_ask_score": 0.9,
        "description": "a dog walking below the eagle",
        "entity_type": "explicit",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "cat",
        "importance_to_ask_score": 0,
        "description": "a cat",
        "entity_type": "explicit",
        "probability_of_appearing": 0
      }},
      {{
        "name": "camera angle",
        "importance_to_ask_score": 0.8,
        "description": "the camera angle of the image",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "cloud",
        "importance_to_ask_score": 0.8,
        "description": "the cloud in the sky where the eagle is soaring",
        "entity_type": "implicit",
        "probability_of_appearing": 0.7
      }},
      {{
        "name": "mountain",
        "importance_to_ask_score": 0.3,
        "description": "the mountain in the background where the eagle is flying",
        "entity_type": "background",
        "probability_of_appearing": 0.1
      }},
      {{
        "name": "trees",
        "importance_to_ask_score": 0.5,
        "description": "trees in the background",
        "entity_type": "background",
        "probability_of_appearing": 0.2
      }},
      {{
        "name": "time of day",
        "importance_to_ask_score": 1.0,
        "description": "the time of the day for the scenario",
        "entity_type": "background",
        "probability_of_appearing": 1.0,
      }},
      {{
        "name": "what the dog is walking on",
        "importance_to_ask_score": 1.0,
        "description": "what the dog is walking on",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "image style",
        "importance_to_ask_score": 1.0,
        "description": "the style of the image",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }}
  ]

  Example3:
  Input: {{
    "user_prompt": "a table in an office"
  }}
  Output: [
      {{
        "name": "table",
        "importance_to_ask_score": 0.8,
        "description": "a table in the image",
        "entity_type": "explicit",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "office",
        "importance_to_ask_score": 0.8,
        "description": "the office where the table is",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "camera angle",
        "importance_to_ask_score": 0.8,
        "description": "the camera angle of the image",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "chair",
        "importance_to_ask_score": 0.7,
        "description": "a chair next to the table",
        "entity_type": "implicit",
        "probability_of_appearing": 0.5
      }},
      {{
        "name": "laptop",
        "importance_to_ask_score": 0.7,
        "description": "a laptop on the table",
        "entity_type": "implicit",
        "probability_of_appearing": 0.5
      }},
      {{
        "name": "books",
        "importance_to_ask_score": 0.6,
        "description": "books on the table",
        "entity_type": "implicit",
        "probability_of_appearing": 0.3
      }},
      {{
        "name": "papers",
        "importance_to_ask_score": 0.5,
        "description": "papers on the table",
        "entity_type": "implicit",
        "probability_of_appearing": 0.5
      }},
      {{
        "name": "people",
        "importance_to_ask_score": 0.7,
        "description": "people working at the office",
        "entity_type": "implicit",
        "probability_of_appearing": 0.3
      }},
      {{
        "name": "coffee mug",
        "importance_to_ask_score": 0.7,
        "description": "a coffee mug on the table",
        "entity_type": "implicit",
        "probability_of_appearing": 0.3
      }},
      {{
        "name": "time of day",
        "importance_to_ask_score": 1.0,
        "description": "the time of the day for the scenario",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }},
      {{
        "name": "image style",
        "importance_to_ask_score": 0.9,
        "description": "the style of the image",
        "entity_type": "background",
        "probability_of_appearing": 1.0
      }}
  ]

  Identify the entities given the input given below. Strictly stick to the format.
  Input: {{
    "user_prompt": "{user_prompt}"
  }}
  Output: """)


def relation_parser_prompt(user_prompt: str, entity_names: List[str]) -> str:
  return textwrap.dedent(f"""\
        Given a text-to-image prompt and a list of entity described in the prompt, your goal is to identify a list of entity pairs that have relations between them. Ignore entity pairs without relations. The output should be a json parse-able format (No comma after the last element of the list):

        Input:
        user_prompt: the prompt from the user.
        entities: a list of entities mentioned in the user_prompt.

        Output:
        name (str): The name of the relation. Use `entity1-entity2` as the format.
        description (str): A short description of the relation.
        spatial_relation (map from potential relation candidates to probability): Possible spatial relations between the two entities. If a relation is mentioned in the user prompt, assign 1.0 as the probability. The sum of probabilities over all relation candidates shall be 1.
        importance_to_ask_score (float): The importance score of asking a question regarding this relation to reduce entropy. This is a number between 0 and 1, higher means more important. Assign a higher score if the two entities are very important, the relation between them is very unclear, and the relation is very important for the layout of the image.
        name_entity_1 (str): The name of the first entity.
        name_entity_2 (str): The name of the second entity.
        is_bidirectional (bool): Whether the relation is bidirectional.

        Below is an example input and output pair:
        Example1:
        Input: {{
          "user_prompt": "generate an image of a lionhead rabbit sitting on grass, and a eagle is flying through the sky. There is a tree in the background.",
          "entity": ["rabbit", "grass", "eagle", "tree"]
        }}
        Output: [
            {{
              "name": "rabbit-grass",
              "description": "rabbit sitting on grass",
              "spatial_relation": {{"above": 0.8, "below": 0.0, "in front of": 0.0, "behind": 0.0, "left of": 0.1, "right of": 0.1}},
              "importance_to_ask_score": 0.1,
              "name_entity_1":"rabbit",
              "name_entity_2": "grass",
              "is_bidirectional": true
            }},
            {{
              "name": "eagle-grass",
              "description": "eagle is flying through the sky",
              "spatial_relation": {{"above": 1.0, "below": 0.0, "in front of": 0.0, "behind": 0.0,"left of": 0.0, "right of": 0.0}},
              "importance_to_ask_score": 0.1,
              "name_entity_1":"eagle",
              "name_entity_2": "grass",
              "is_bidirectional": false
            }},
            {{
              "name": "tree-grass",
              "description": "",
              "spatial_relation": {{"above": 0.5, "below": 0.0, "in front of": 0.0, "behind": 0.0, "left of": 0.25, "right of": 0.25}},
              "importance_to_ask_score": 0.1,
              "name_entity_1":"tree",
              "name_entity_2": "grass",
              "is_bidirectional": false
            }},
            {{
              "name": "eagle-rabbit",
              "description": "",
              "spatial_relation": {{"above": 0.8, "below": 0.0, "in front of": 0.0, "behind": 0.0, "left of": 0.1, "right of": 0.1}},
              "importance_to_ask_score": 0.1,
              "name_entity_1":"eagle",
              "name_entity_2": "rabbit",
              "is_bidirectional": false
            }},
            {{
              "name": "tree-rabbit",
              "description": "",
              "spatial_relation": {{"above": 0.2, "below": 0.0, "in front of": 0.1, "behind": 0.5, "left of": 0.1, "right of": 0.1}},
              "importance_to_ask_score": 0.8,
              "name_entity_1":"tree",
              "name_entity_2": "rabbit",
              "is_bidirectional": false
            }},
            {{
              "name": "tree-eagle",
              "description": "",
              "spatial_relation": {{"above": 0.1, "below": 0.5, "in front of": 0.1, "behind": 0.1, "left of": 0.1, "right of": 0.1}},
              "importance_to_ask_score": 0.7,
              "name_entity_1":"tree",
              "name_entity_2": "eagle",
              "is_bidirectional": false
            }},
          ]

        Identify relationships between entities given the input given below. Strictly stick to the format.
        Input: {{
          "user_prompt": "{user_prompt}",
          "entity": "{entity_names}"
        }}
        Output: """)


def attribute_parser_prompt(
    user_prompt: str, entity: Entity, entity_list: List[Entity]
) -> str:
  existing_entities = [
      ent.name for ent in entity_list if ent.name != entity.name
  ]
  existing_entities = ', '.join(existing_entities)
  return textwrap.dedent(f"""\
    Given a text-to-image prompt and a particular entity described in the prompt, and your goal is to identify a list possible attributes that could describe the particular entity. Output Requirements:

    1. if this attribute has already existed as an entity in other existing entity list, then do not include it.
    2. the attribute candidate could be a mixed of values like `color A and color B`.
    3. The output should be a json parse-able format:

    name (str): The name of the attribute.
    importance_to_ask_score (float): The importance score of asking a question about this attribute to reduce the uncertainty of what the image is given the user prompt. This is a number between 0 and 1, higher means more important. Consider these factors when assigning scores: 1. Increate the score for attributes that are the primary attributes of an important entity; 2. significantly increase the score for attributes that could strongly influence the generation or portrayal of OTHER attributes in the scene; 3. descrease the score for attributes that are already well specified in the prompt. For example, a breed of a dog would impact other attributes like color, size, etc. So the breed attribute should have a higher importance score than color, size, etc. Assign a much lower score if the attribute's value is already mentioned in the user prompt.
    candidates (List of names and probabilities): List of possible values that the attribute can take. Make sure to generate atleast 5 or more possible values. These should be realistic for the given entity. For each attribute, returns the probability that the user wants this candidate based on the user prompt. If it's already mentioned by the user, only generate one candidate (the mentioned one) and assign 1.0 as the probability. The sum of probabilities over all candidates shall be 1. Also infer the probability based on the prompt. For example, for a dog with breed Samoyed, the color attribute has a very high probability of white.

    Below are two examples of input and output pairs:

    Example 1:
    Input: {{
      "user_prompt": "generate an image of a white rabbit running on grass",
      "entity": "rabbit",
      "other_existing_entities": "grass"
    }}
    Output: [
        {{
          "name": "color",
          "importance_to_ask_score": 0.9,
          "candidates": {{"white":1.0}}
        }},
        {{
          "name": "breed",
          "importance_to_ask_score": 1.0,
          "candidates": {{"Dutch": 0.20,
                        "Mini Lop": 0.15,
                        "Netherland Dwarf": 0.15,
                        "Lionhead": 0.10,
                        "Flemish Giant": 0.10,
                        "Mini Rex": 0.10,
                        "English Angora": 0.08,
                        "Mini Satin": 0.05,
                        "Himalayan": 0.05,
                        "Californian": 0.02}}
        }},
        {{
          "name": "age",
          "importance_to_ask_score": 0.1,
          "candidates": {{"adult": 0.6,
                          "baby": 0.2,
                          "senior": 0.2}}
        }}
      ]

    Example 2:
    Input: {{
      "user_prompt": "a heart-shaped cake decorated with red and blue flowers",
      "entity": "cake",
      "other_existing_entities": "flowers"
    }}
    Output: [
        {{
          "name": "color",
          "importance_to_ask_score": 1.0,
          "candidates": {{"red and blue": 0.5,
                          "red": 0.2,
                          "pink": 0.05,
                          "blue": 0.2,
                          "green": 0.05}}
        }},
        {{
          "name": "size",
          "importance_to_ask_score": 0.8,
          "candidates": {{"large": 0.25,
                        "medium-sized": 0.25,
                        "small": 0.25,
                        "mini": 0.025}}
        }},
        {{
          "name": "shape",
          "importance_to_ask_score": 0.6,
          "candidates": {{"heart": 1.0}}
        }}
      ]

    Generate attributes given the input given below. Do not include other entities in the attributes. Strictly stick to the format.
    Input: {{
      "user_prompt": "{user_prompt}",
      "entity": "{entity.name}",
      "other_existing_entities": "{existing_entities}"
    }}
    Output: """)


def attribute_parser_prompt_w_background(
    user_prompt: str, entity: Entity, entity_list: List[Entity]
) -> str:
  existing_entities = [
      ent.name for ent in entity_list if ent.name != entity.name
  ]
  existing_entities = ', '.join(existing_entities)
  return textwrap.dedent(f"""\
    Given a text-to-image prompt and a particular background entity described in the prompt, your goal is to identify a list possible values for the background entity. Note that, if this attribute has already existed as an entity in other existing entity list, then do not include it. The output should be a json parse-able format:

    name (str): the same as the entity name.
    importance_to_ask_score (float): The importance score of asking a question about this attribute to reduce the uncertainty of what the image is given the user prompt. This is a number between 0 and 1, higher means more important. Consider these factors when assigning scores: 1. Increate the score for attributes that are the primary attributes of an important entity; 2. significantly increase the score for attributes that could strongly influence the generation or portrayal of OTHER attributes in the scene; 3. descrease the score for attributes that are already well specified in the prompt. For example, a breed of a dog would impact other attributes like color, size, etc. So the breed attribute should have a higher importance score than color, size, etc. Assign a much lower score if the attribute's value is already mentioned in the user prompt.
    candidates (List of names and probabilities): List of possible values that the attribute can take. These should be realistic for the given entity. For each attribute, returns the probability that the user wants this candidate based on the user prompt. If it's already mentioned by the user, only generate one candidate (the mentioned one) and assign 1.0 as the probability. The sum of probabilities over all candidates shall be 1.

    Below is an example input and output pair:
    Example1:
    Input: {{
      "user_prompt": "generate an image of a white lionhead rabbit running on grass",
      "entity": "weather",
      "other_existing_entities": "grass, rabbit"
    }}
    Output: [
        {{
          "name": "weather",
          "importance_to_ask_score": 1.0,
          "candidates": {{"sunny":0.7, "rainy": 0.1, "cloudy": 0.2}}
        }}
      ]

    Example2:
    Input: {{
      "user_prompt": "An eagle soars through the skies while a dog walks below",
      "entity": "time of day",
      "other_existing_entities": "eagle, sky, dog"
    }}
    Output: [
        {{
          "name": "time of day",
          "importance_to_ask_score": 1.0,
          "candidates": {{"sunset":0.3, "sunrise": 0.25, "morning": 0.2, "night": 0.05, "afternoon": 0.2}}
        }}
      ]

    Generate attributes given the input given below. Strictly stick to the format.
    Input: {{
      "user_prompt": "{user_prompt}",
      "entity": "{entity.name}",
      "other_existing_entities": "{existing_entities}"
    }}
    Output:""")


def agent_history_to_conversation(history: list[dict[str, Any]]) -> str:
  """Converts agent history to a conversation.

  Args:
    history: history of the agent.

  Returns:
    Conversation based on the history.
  Raises:
    ValueError: agent action is of unknown type.
  """
  if not history:
    return ''
  conversation = 'User: ' + history[0]['prompt']
  for history_item in history:
    if isinstance(history_item['action'], str):
      action = history_item['action']
    elif history_item['action'] is None:
      action = ''
    else:
      action = history_item['action'].verbalized_action
    observation = history_item['observation']
    conversation = '\n'.join(
        [conversation, 'Agent: ' + action, 'User: ' + observation]
    )
  return conversation


def select_action_based_on_belief_prompt(
    user_prompt: str,
    belief_state: BeliefState,
    history: list[dict[str, Any]],
) -> str:
  """Prompt for select action for BeliefAgent.

  Args:
    user_prompt: The prompt of the user.
    belief_state: Belief state of the agent.
    history: History of the agent.

  Returns:
    Prompt for select action.
  """
  conversation = agent_history_to_conversation(history)
  return textwrap.dedent(f"""\
You are an intelligent agent that helps users generate images. \
Before generating the image requested by the user, you should ask the most \
important clarification questions to make sure you understand the key features of the image.
The user describes the image as: {user_prompt}.
The following is your belief of what the image contains, including the entities, attributes of each entity and relations between entities.
Each entity has "name", "descriptions", "importance to ask score" and  "probability of appearing". \
"Name" is the identifier of the entity. "Descriptions" is the description of the entity. \
"Importance to ask score" is how important it is for the agent to ask whether the entity exists. \
"Probability of appearing" is the probability the agent estimated that this entity exits in the image.

Each entity has a list of attributes. Each attribute has "name", "importance to ask score" and "candidates". \
"Name" is the identifier of the attribute. "Importance to ask score" is how important it is to ask about the exact value for the attribute of the entity. \
"Candidates" is a list of possible values for the attribute.

Each candidate value has a probability that describes how likely this candidate value should be assigned to the attribute.
For example, "Attribute Name: color, Importance to ask Score: 0.9, Candidates: [white: 0.5, black: 0.5]" means the color is either white or black, each with 0.5 probability. \
If you ask about attributes, you should ask about the attribute with the highest uncertainty. Your uncertainty can be judged by the probabilities. If the probabilities are 0.5 and 0.5, you are uncertain. \
If the probabilities are 0.1 and 0.9, you are fairly certain.

The agent belief is:
{belief_state.__str__()}

Based on the user prompt "{user_prompt}" and the belief of the agent, please provide a question to ask about the image. \
The question should be as concise and direct as possible. The question should aim to obtain the most information about the style, entities, attributes, spatial layout and other contents of the image. \
Remember to ask for information that are critical to knowing the critical details of the image that is important to the user. \
The question should reduce your uncertainty about the user intent as much as possible. \
DO NOT ask question that can be answered by common sense. \
DO NOT ask question that are obvious to answer based on the user prompt "{user_prompt}". \
DO NOT ask any question about information present in the following user-agent dialogue within <dialogue> and </dialogue> markers.

<dialogue>
{conversation}
</dialogue>

DO NOT ask any question that has been asked in the dialogue above.

Your question does not have to be entirely decided by the belief. You can construct any question that make yourself more confident about what the image is.
Think step by step and reason about your uncertainty of the image to generate. Make sure to ask only one question. Make sure it is not very difficult for the user to answer. For example, do not ask a very very long question, which can take the user a long time to read and answer.
Make sure that you question the answer within <question> and </question> markers.
""")


def simulated_random_response_user_prompt(
    user_prompt: str, agent_question: str, belief_state: BeliefState
):
  """Prompt for simulated random response to an agent question.

  Args:
    user_prompt: The prompt of the user.
    agent_question: The question asked by the agent.
    belief_state: The state of the agent.

  Returns:
    Prompt for simulated random response to an agent question.
  """
  return f"""\
You are a user using an intelligent text-to-image agent to generate an image with the prompt "{user_prompt}". \
Because your prompt does not provide enough information, the agent asks you a question "{agent_question}". \
You should answer the question by drawing a random sample from what you believe the image should be.

The following is your belief of what the image contains, including the entities, attributes of each entity and relations between entities.
Each entity has "name", "descriptions", "importance to ask score" and  "probability of appearing". \
"Name" is the identifier of the entity. "Descriptions" is the description of the entity. \
"Importance to ask score" is how important this entity is for the image you have in mind. \
"Probability of appearing" is the probability the agent estimated that this entity exits in the image.
If the agent asks about the existence of an entity and the probability of appearing is 0.8, you should answer yes with 0.8 probability and no with 0.2 probability.

Each entity has a list of attributes. Each attribute has "name", "importance to ask score" and "candidates". \
"Name" is the identifier of the attribute. "Importance to ask score" is how important it is to ask about the exact value for the attribute of the entity. \
"Candidates" is a list of possible values for the attribute. \
Each candidate value has a probability that describes how likely this candidate value should be assigned to the attribute.
For example, the entity "rabbit" has an attribute "color" and the attribute might be "Attribute Name: color, Importance to ask Score: 0.9, Candidates: [white: 0.5, black: 0.5]", which means the color is either white or black, each with 0.5 probability. \
In this case, if the agent asks "What is the color of the rabbit?", you should answer "White" with 0.5 probability and "Black" with 0.5 probability. Do not vaguely answer "It can be either white or black."

Your belief is:
{belief_state.__str__()}


Answer the following question "{agent_question}" by drawing a random sample from what you believe the image should be.
Please answer the question confidently, as if you have seen the image. DO NOT be vague. DO NOT answer "I don't know". If you don't know, answer with the most likely answer based on common sense.
Make sure that your answer is within <answer> and </answer> markers. Think step by step and reason about how to draw a random sample from what you believe the image should be.
"""


def simulated_belief_user_prompt(
    long_user_prompt: str, agent_question: str, user_belief_state: BeliefState
):
  """Prompt for simulated random response to an agent question.

  Args:
    long_user_prompt: The prompt of the user.
    agent_question: The question asked by the agent.
    user_belief_state: The state of the agent.

  Returns:
    Prompt for simulated random response to an agent question.
  """
  return f"""\
You are a user using an intelligent text-to-image agent to generate an image described by "{long_user_prompt}". \
The agent does not have access to the entire image description and asks you a question "{agent_question}". \
You should answer the question based on the image description and your belief of the rest of the details of the image.
Your answer must be faithful to the image description "{long_user_prompt}". If the image description does not provide enough information to answer the question, you should answer by inferring the most likely answer based on your belief.

The following is your belief of what the image contains, including the entities, attributes of each entity and relations between entities.
Each entity has "name", "descriptions", "importance to ask score" and  "probability of appearing". \
"Name" is the identifier of the entity. "Descriptions" is the description of the entity. \
"Importance to ask score" is how important this entity is for the image you have in mind. If this entity is very important to you (meaning that you want every detail of this entity to be correct), you should indicate that this entity is important. \
"Probability of appearing" is the probability the agent estimated that this entity exits in the image.
If the agent asks about the existence of an entity that does not exist in the image description, and the probability of appearing larger than 0.5, you should answer yes since this entity is likely to appear in the image. Otherwise, if the probability of appearing less than or equal to 0.5, you should answer no.

Each entity has a list of attributes. Each attribute has "name", "importance to ask score" and "candidates". \
"Name" is the identifier of the attribute. "Importance to ask score" is how important it is to ask about the exact value for the attribute of the entity. \
"Candidates" is a list of possible values for the attribute. \
Each candidate value has a probability that describes how likely this candidate value should be assigned to the attribute.
For example, the entity "rabbit" has an attribute "color" and the attribute might be "Attribute Name: color, Importance to ask Score: 0.9, Candidates: [white: 0.5, black: 0.5]", which means the color is either white or black, each with 0.5 probability. \
In this case, if the color is not mentioned in the image description and the agent asks "What is the color of the rabbit?", you should answer "White" since it is the first item in the list of candidates. Do not vaguely answer "It can be either white or black."

Your belief is:
{user_belief_state.__str__()}


Answer the following question "{agent_question}" based on the image description "{long_user_prompt}". If the image description does not provide enough information to answer the question, you should answer by inferring the most likely answer based on your belief.
Your answer must be within <answer> and </answer> markers.
Please answer the question confidently, as if you have seen the image. DO NOT be vague. DO NOT answer "I don't know". If you don't know, answer with the most likely answer based on common sense.
Think step by step and reason about how to best answer the question based on the image description. If the image description does not contain the information to answer the question, reason about the most likely answer based on what you believe the image should be.
Remember, your answer must be faithful to the image description. You must answer as if you have seen all the details of the image.
Remember to answer the question within <answer> and </answer> markers. Be confident and be concise.
"""
