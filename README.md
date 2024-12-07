# Proactive Agents for Multi-Turn Text-to-Image Generation under Uncertainty


User prompts for generative AI models are often underspecified or open-ended,
which may lead to sub-optimal responses. This prompt underspecification problem
is particularly evident in text-to-image (T2I) generation, where users commonly
struggle to articulate their precise intent. This disconnect between the user’s vision
and the model’s interpretation often forces users to painstakingly and repeatedly
refine their prompts. To address this, we propose a design for proactive T2I agents
equipped with an interface to actively ask clarification questions when uncertain,
and present their understanding of user intent as an interpretable belief graph
that a user can edit. We build simple prototypes for such agents and verify their
effectiveness through both human studies and automated evaluation. We observed
that at least 90% of human subjects found these agents and their belief graphs
helpful for their T2I workflow. Moreover, we use a scalable automated evaluation
approach using two agents, one with a ground truth image and the other tries to
ask as few questions as possible to align with the ground truth. On DesignBench, a
benchmark we created for artists and designers, the COCO dataset (Lin et al., 2014)
and ImageInWords (Garg et al., 2024), we observed that these T2I agents were able
to ask informative questions and elicit crucial information to achieve successful
alignment with at least 2 times higher VQAScore (Lin et al., 2024) than the standard
single-turn T2I generation. 


[Demo Video](https://www.youtube.com/watch?v=HQgjLWp4Lo8) , [Paper Link](https://openreview.net/pdf?id=xsmlrhoQzC)

![Alt Text](Fig.png)

## Stay Tuned for Code Release!

## License


Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

