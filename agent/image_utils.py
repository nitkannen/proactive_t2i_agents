"""Utility functions for working with images."""

import matplotlib.pyplot as plt
import PIL.Image

from agent import agent_classes as ac
from agent import data_classes as dc
from agent import llm_parser as parser
from agent import utility_classes as util

def load_img(path):
  return PIL.Image.open(path)

def show_image(image_pil):
  return plt.imshow(image_pil)

def get_image_description(image_path: str) -> str:
  image_pil = load_img(image_path)
  image_bytes = image_pil.tobytes()
  image_bytes_str = str(image_bytes)
  return image_bytes_str


def get_scene_graph_from_image(
    image_path: str,
    entity_parser: parser.EntityParser,
    relation_parser: parser.RelationParser,
    attribute_parser: parser.AttributeParser,
) -> dc.BeliefState:
  """Generates a scene graph representing the entities, relations, and attributes within an image.

  Args:
      image_path: The path to the image file.
      entity_parser: An EntityParser instance
      relation_parser: A RelationParser instance
      attribute_parser: An AttributeParser instance

  Returns:
      A SceneGraph object

  Example usage:
      scene_graph =
      get_scene_graph_from_image('qfa/images/beagle_cat_image.png',
      entity_parser,
      relation_parser, attribute_parser)
  """
  image_pil = load_img(image_path)
  # get a highly descriptive text description of the image
  prompt = """
    Provide a highly detailed description of the image. Make sure you identify all the entities in the image and describe the entities and their attributes.
    An entity is a real world object. An attribute is a property of an entity.
    Some example entity, attribute pairs are (tree, type), (dog, breed), (cat, color), (mountain, size), (river, color), (fruit, name). Describe all details about the entities.
    Next also describe the spatial relationship between the different entities in the image. Provide throrough details about the scene, color, size etc.
    Write a textual description of the image with the above instructions
    """
  vlm = util.VLM()
  description = vlm.generate(image_pil, prompt)

  scene_graph = ac.get_belief_state_from_prompt(
      description, entity_parser, relation_parser, attribute_parser
  )

  return scene_graph
