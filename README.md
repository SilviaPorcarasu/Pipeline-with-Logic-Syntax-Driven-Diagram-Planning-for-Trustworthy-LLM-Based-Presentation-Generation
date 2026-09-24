# Pipeline with Logic-Syntax-Driven Diagram Planning for Trustworthy LLM-Based Presentation Generation

The pipeline generates educational presentations from a topic and retrieved source material. It combines retrieval-augmented slide planning, logic and syntax driven diagram planning through Prolog representations, D2 diagram rendering, and multi-stage verification.

Three diagram planning strategies are investigated:
- Direct LLM-based planning.
- Planning guided by Prolog facts and rules extracted using syntactic analysis.
- Planning guided by LLM-generated Prolog facts and rules.

## Paper

The conference manuscript is available [here](paper/Pipeline_with_Logic_Syntax_Driven_Diagram_Planning_for_Trustworthy_LLM_Based_Presentation_Generation.pdf).

Publication details will be added when available

## Requirements

- Python 3.11+
- Node.js and npm
- D2 CLI for diagram rendering
- Marp CLI and Chrome/Chromium for PowerPoint export
- An API key for the selected LLM provider

