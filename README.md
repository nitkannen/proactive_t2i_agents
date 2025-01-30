# [Proactive Agents for Multi-Turn Text-to-Image Generation under Uncertainty]


User prompts for generative AI models are often underspecified, leading to a misalignment between the user intent and models' understanding. As a result, users commonly have to painstakingly refine their prompts again and again. We study this problem in text-to-image (T2I) generation and propose a prototype for proactive T2I agents equipped with an interface to (1) actively ask clarification questions when uncertain, and (2) present their uncertainty about user intent as an understandable and editable belief graph. We build simple prototypes for such agents and propose a new scalable and automated evaluation approach using two agents, one with a ground truth intent (an image) and the other tries to ask as few questions as possible to align with the ground truth. On DesignBench, a benchmark we created for artists and designers, the COCO dataset, and ImageInWords, we observed that these T2I agents were able to ask informative questions and elicit crucial information to achieve successful alignment with at least 2 times higher VQAScore than the standard T2I generation. Moreover, we conducted human studies and observed that at least 90% of human subjects found these agents and their belief graphs helpful for their T2I workflow, highlighting the effectiveness of our approach.


[Demo Video](https://www.youtube.com/watch?v=HQgjLWp4Lo8)

![Alt Text](figures/Fig.png)

## Proactive Agent code

The ```agent/``` directory contains the implementation of the proposed agent that is powered by Gemini 1.5 Pro.


## DesignBench Dataset

### Data Info
The dataset includes 30 images containing different objects and scenes. 

The images have been sourced from www.unsplash.com, www.pexels.com, www.freepik.com (the licenses for the images are listed below).

Out of the 30 images: 8  contain animals, 9 images contain humans or partial human figures, 15 images contain only inanimate objects and 2 contain only a scene. The dataset contains a variable number of subjects (1-8) per image. Images are captured in different conditions, environments and under different angles.We include a file dataset/prompts_and_classes.json which contains two types of prompts per image: a lengthy detailed prompt and a short concise prompt lacking details. These are the prompts used in the paper for all experiments using DesignBench. The images have been sourced from www.unsplash.com, www.pexels.com, www.freepik.com.

### Data Access

`designbench/prompts_and_classes.json` file contains a list of all the image names, reference links to the images, and a short and long prompt per image. The images have been cropped from their original form to directly download the cropped version of the photos that was used in the paper visit `designbench/images/` 

### Licenses
**Unsplash**: 
Unsplash grants you an irrevocable, nonexclusive, worldwide copyright license to download, copy, modify, distribute, perform, and use images from Unsplash for free, including for commercial purposes, without permission from or attributing the photographer or Unsplash. This license does not include the right to compile images from Unsplash to replicate a similar or competing service.

**Pexels**: All photos and videos uploaded on Pexels are licensed under the Pexels license. This means you can use them for free for personal and commercial purposes without attribution. For more information read the following questions in this guide, or refer to our license page or our Terms of Service.

**FreePik**: Freepik and Flaticon PDF licenses allow you to use our resources without crediting the author and these will remain active even when your Premium or Premium+ subscription has expired.Please remember to download the license PDF files and keep them in a safe place. We recommend you to download them right after downloading the file.https://support.freepik.com/s/article/How-to-download-Freepik-premium-licenses?language=en_US

```
## License

Copyright 2024 Proactive Agents T2I Authors

