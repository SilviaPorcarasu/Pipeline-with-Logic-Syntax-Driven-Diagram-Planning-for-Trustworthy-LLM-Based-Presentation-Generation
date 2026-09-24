"""Prompt templates for multi-step presentation generation.

Pipeline:
  Prompt 1   — global_plan_prompt: plan global cu idee + tema diagrama per slide
  Per slide (cu planul global in context):
    Prompt 2.1 — slide_detail_prompt: detalii slide + tema diagrama
    Prompt 2.2 — diagram_concept_prompt: concept diagrama (noduri, sageti, flow)
    Prompt 2.3 — (in llm_service.py) _format_diagram_code_with_gemini_d2: cod D2
"""

from __future__ import annotations

import os

from models import PresentationRequest, SlideOutline



# BATCH MODE — One mega-prompt that generates the entire deck at once.


# Reference D2 few-shots included in the batch prompt so Gemini can mimic
# rich visual style with containers, named colors, labeled edges, shapes.
_BATCH_FEW_SHOTS = [
    (
        "Transformer with encoder/decoder containers and cross-attention bridge",
        "direction: right\n"
        "encoder: Encoder Stack {\n"
        '  style.fill: "aliceblue"\n'
        "  style.stroke-dash: 3\n"
        "  emb: Input Embedding { shape: rectangle }\n"
        "  mha: Multi-Head Attention { shape: hexagon }\n"
        "  ff: Feed Forward { shape: rectangle }\n"
        "  norm: Layer Norm { shape: diamond }\n"
        "  emb -> mha: tokens\n"
        "  mha -> ff: attended\n"
        "  ff -> norm: residual\n"
        "  emb -> norm: skip\n"
        "}\n"
        "decoder: Decoder Stack {\n"
        '  style.fill: "peachpuff"\n'
        "  style.stroke-dash: 3\n"
        "  masked: Masked Attention { shape: hexagon }\n"
        "  cross: Cross Attention { shape: hexagon }\n"
        "  ff2: Feed Forward { shape: rectangle }\n"
        "  out: Output Probs { shape: diamond }\n"
        "  masked -> cross: queries\n"
        "  cross -> ff2: context\n"
        "  ff2 -> out: logits\n"
        "}\n"
        "encoder.norm -> decoder.cross: keys + values\n"
        "style.font-size: 32"
    ),
    (
        "RAG pipeline with retrieval and generation stages, feedback loop",
        "direction: down\n"
        "user: User Query { shape: oval; style.fill: \"aliceblue\" }\n"
        "retrieval: Retrieval Stage {\n"
        '  style.fill: "lavenderblush"\n'
        "  style.stroke-dash: 3\n"
        "  enc: Encoder { shape: rectangle }\n"
        "  idx: Vector Index { shape: cylinder }\n"
        "  kb: Knowledge Base { shape: cylinder }\n"
        "  enc -> idx: embed\n"
        "  kb -> idx: indexed\n"
        "}\n"
        "generation: Generation Stage {\n"
        '  style.fill: "honeydew"\n'
        "  ctx: Context Builder { shape: rectangle }\n"
        "  llm: LLM Generator { shape: hexagon }\n"
        "  ctx -> llm: prompt\n"
        "}\n"
        "answer: Answer { shape: oval; style.fill: \"lemonchiffon\" }\n"
        "user -> retrieval.enc: question\n"
        "retrieval.idx -> generation.ctx: top-k chunks\n"
        "generation.llm -> answer: response\n"
        "answer -> user: feedback\n"
        "style.font-size: 32"
    ),
    (
        "Training loop with data, model, evaluation branches",
        "direction: right\n"
        "data: Data Pipeline {\n"
        '  style.fill: "mintcream"\n'
        "  raw: Raw Data { shape: cylinder }\n"
        "  clean: Preprocess { shape: rectangle }\n"
        "  split: Train/Test { shape: diamond }\n"
        "  raw -> clean: load\n"
        "  clean -> split: validate\n"
        "}\n"
        "training: Training {\n"
        '  style.fill: "papayawhip"\n'
        "  style.stroke-dash: 3\n"
        "  model: Model { shape: hexagon }\n"
        "  eval: Evaluate { shape: diamond }\n"
        "  tune: Tune { shape: oval }\n"
        "  model -> eval: predict\n"
        "  eval -> tune: if poor\n"
        "  tune -> model: retrain\n"
        "}\n"
        "deploy: Deploy { shape: hexagon; style.fill: \"honeydew\" }\n"
        "data.split -> training.model: train set\n"
        "data.split -> training.eval: test set\n"
        "training.eval -> deploy: if good\n"
        "style.font-size: 32"
    ),
    (
        "Classification metrics: predictions, confusion matrix, final scores",
        "direction: right\n"
        "inputs: Model Outputs {\n"
        '  style.fill: "lavender"\n'
        "  pred: Predictions { shape: rectangle }\n"
        "  truth: True Labels { shape: rectangle }\n"
        "}\n"
        "confusion: Confusion Matrix {\n"
        '  style.fill: "mistyrose"\n'
        "  style.stroke-dash: 3\n"
        "  tp: True Positives { shape: diamond }\n"
        "  fp: False Positives { shape: oval }\n"
        "  fn: False Negatives { shape: oval }\n"
        "  tn: True Negatives { shape: diamond }\n"
        "}\n"
        "metrics: Metrics {\n"
        '  style.fill: "honeydew"\n'
        "  prec: Precision { shape: hexagon }\n"
        "  rec: Recall { shape: hexagon }\n"
        "  f1: F1 Score { shape: cylinder }\n"
        "  prec -> f1: harmonic\n"
        "  rec -> f1: mean\n"
        "}\n"
        "inputs.pred -> confusion.tp: correct\n"
        "inputs.pred -> confusion.fp: wrong\n"
        "inputs.truth -> confusion.fn: missed\n"
        "confusion.tp -> metrics.prec\n"
        "confusion.fp -> metrics.prec\n"
        "confusion.tp -> metrics.rec\n"
        "confusion.fn -> metrics.rec\n"
        "style.font-size: 32"
    ),
    (
        "Rich experiment platform: vars, sketch, multiple instances, animated edges, cross-container refs",
        "vars: {\n"
        "  d2-config: {\n"
        "    theme-id: 3\n"
        "    sketch: true\n"
        "    layout-engine: elk\n"
        "  }\n"
        "  colors: {\n"
        '    c2: "#C7F1FF"\n'
        '    c3: "#B5AFF6"\n'
        '    c4: "#DEE1EB"\n'
        '    c5: "#88DCF7"\n'
        '    c6: "#E4DBFE"\n'
        "  }\n"
        "}\n"
        "\n"
        "LangUnits: {\n"
        "  style.fill: ${colors.c6}\n"
        "  RegexVal\n"
        "  SQLSelect\n"
        "  PythonTr\n"
        "  langunit_n: {\n"
        "    style.multiple: true\n"
        "    style.stroke-dash: 10\n"
        "    style.stroke: black\n"
        "    style.animated: 1\n"
        "  }\n"
        "}\n"
        "\n"
        "DatasetUI: {\n"
        "  style.fill: ${colors.c4}\n"
        "}\n"
        "\n"
        "ExperimentHost: {\n"
        "  style.fill: ${colors.c4}\n"
        "  Experiment: {\n"
        "    style.multiple: true\n"
        "  }\n"
        "  Dataset\n"
        "}\n"
        "\n"
        "ModelConfiguration: {\n"
        "  style.fill: ${colors.c2}\n"
        "  Prompting\n"
        "  Model\n"
        "  LangUnit\n"
        "}\n"
        "\n"
        'LangUnits <- ExperimentHost.Dataset: "load dataset"\n'
        'DatasetUI -> LangUnits: "manage datasets"\n'
        "ExperimentHost.Experiment -> ModelConfiguration\n"
        "ExperimentHost.Experiment.ModelConfigurations -> ModelConfiguration: configure\n"
        "style.font-size: 32"
    ),
    (
        "Resource graph with markdown title, tooltips, grid layout, wildcard edge styles",
        "vars: {\n"
        "  d2-config: {\n"
        "    layout-engine: elk\n"
        "  }\n"
        "}\n"
        "\n"
        "*.style.font-size: 22\n"
        "*.*.style.font-size: 22\n"
        "\n"
        "title: |md\n"
        "  # Terraform resources (v1.0.0)\n"
        "| {near: top-center}\n"
        "\n"
        "direction: right\n"
        "\n"
        "project_connection: {\n"
        "  style: {\n"
        '    fill: "#C5C6C7"\n'
        "    stroke: grey\n"
        "  }\n"
        "}\n"
        "\n"
        "privatelink_endpoint: {tooltip: Datasource only}\n"
        "group\n"
        "group_partial_permissions\n"
        "service_token\n"
        "job: {\n"
        "  style: {\n"
        '    fill: "#ACE1AF"\n'
        "    stroke: green\n"
        "  }\n"
        "}\n"
        "\n"
        "conns: Connections (legacy) {\n"
        "  bigquery_connection\n"
        "  fabric_connection\n"
        "  connection\n"
        '  bigquery_connection.style.fill: "#C5C6C7"\n'
        '  fabric_connection.style.fill: "#C5C6C7"\n'
        '  connection.style.fill: "#C5C6C7"\n'
        "}\n"
        'conns.style.fill: "#C5C6C7"\n'
        "\n"
        "env_creds: Environment Credentials {\n"
        "  grid-columns: 2\n"
        "  athena_credential\n"
        "  databricks_credential\n"
        "  snowflake_credential\n"
        "  bigquery_credential\n"
        "  fabric_credential\n"
        "  postgres_credential: {tooltip: Is used for Redshift as well}\n"
        "  teradata_credential\n"
        "}\n"
        "\n"
        "service_token -- project: can scope to {\n"
        "  style: {\n"
        "    stroke-dash: 3\n"
        "  }\n"
        "}\n"
        "group -- project\n"
        "group_partial_permissions -- project\n"
        "user_groups -- group\n"
        "user_groups -- group_partial_permissions\n"
        "project -- environment\n"
        "job -- environment\n"
        "notification -- job\n"
        "webhook -- job: triggered by {\n"
        "  style: {\n"
        "    stroke-dash: 3\n"
        "  }\n"
        "}\n"
        "environment -- global_connection\n"
        "environment -- conns\n"
        "global_connection -- privatelink_endpoint\n"
        "environment -- env_creds\n"
        "project -- project_repository\n"
        "project_repository -- repository\n"
        "environment -- environment_variable\n"
        "\n"
        "project -- project_connection {\n"
        "  style: {\n"
        '    stroke: "#C5C6C7"\n'
        "  }\n"
        "}\n"
        "project_connection -- conns {\n"
        "  style: {\n"
        '    stroke: "#C5C6C7"\n'
        "  }\n"
        "}\n"
        "\n"
        "(job -- *)[*].style.stroke: green\n"
        "(* -- job)[*].style.stroke: green\n"
        "\n"
        'account_level_settings: "Account level settings" {\n'
        "  account_features\n"
        "  ip_restrictions_rule\n"
        "  license_map\n"
        "}\n"
        "account_level_settings.style.fill-pattern: dots"
    ),
]


def batch_deck_prompt(req: PresentationRequest, outline: list[SlideOutline], rag_entries_by_slide: dict[int, str]) -> str:
    """Generate the ENTIRE deck (all slides + all diagrams) in one Gemini call."""
    slides_block = ""
    for s in outline:
        rag_text = rag_entries_by_slide.get(s.slide_number, "")
        slides_block += (
            f"\n--- Slide {s.slide_number} ({s.type}) ---\n"
            f"Title: {s.title}\n"
            f"Goal: {s.goal}\n"
            f"RAG entries:\n{rag_text}\n"
        )

    few_shots_block = "\n".join(
        f"\nEXAMPLE {i + 1} — {name}:\n{code}"
        for i, (name, code) in enumerate(_BATCH_FEW_SHOTS)
    )

    return (
        "Generate a COMPLETE academic presentation deck in one response.\n"
        "For each slide: produce bullets, speech, diagram description, AND RICH D2 code.\n"
        "All slides together — diagrams MUST be visually distinct AND visually rich.\n"
        "\n"
        "REFERENCE D2 EXAMPLES — study these carefully and match their richness:\n"
        f"{few_shots_block}\n"
        "\n"
        f"Topic: {req.topic}\n"
        f"Style: {req.style}\n"
        f"Total slides: {req.slides_count}\n"
        "\nSlide outline with RAG:"
        f"{slides_block}\n"
        "\nReturn JSON with key `slides` — an array with one object per slide.\n"
        "Each slide object must have:\n"
        "- slide_number (int)\n"
        "- main_idea (1 sentence — UNIQUE concept specific to this slide)\n"
        "- teaching_scenario (short paragraph, <= 55 words)\n"
        "- structure_plan (ONE complete sentence directly about the slide's topic, 8-20 words, used as a subtitle; NO meta-text like 'the slide shows' or 'organized by')\n"
        "- bullet_plan (array of 2-3 objects; each `bullet` is a full explanatory sentence 14-22 words that states WHAT happens and WHY/HOW; `explanation` is an extra 12-22 words giving the mechanism or consequence)\n"
        "- diagram_description (format: `nodes: N1 (shape, Label), ... | arrows: ... | direction: right/down | notes: ...`)\n"
        "- diagram_code (valid D2 code — RICH with containers, colors, labeled edges, HUMAN-READABLE labels)\n"
        "- speech_plan (speaker script, <= 110 words, reuse RAG facts)\n"
        "\nRules (strict):\n"
        "- Every slide must have DIFFERENT main_idea and diagram — no repetition.\n"
        "- Progression: slide 1 = overview, middle = specific details, last = summary.\n"
        "- Ground every idea in RAG entries. Each bullet MUST reuse at least 2 exact technical terms from RAG (verbatim).\n"
        "- speech_plan MUST reuse at least 4 exact technical terms from RAG.\n"
        "- Avoid paraphrasing — keep domain terminology literal as in the RAG text.\n"
        "- Professor tone for undergraduate students.\n"
        "\nSUBTITLE (structure_plan) RULES:\n"
        "  - Must be a COMPLETE thought that fits entirely in <= 90 characters.\n"
        "  - Prefer short tight phrasing so nothing gets cut off.\n"
        "  - May end with or without a period — but must not dangle on a preposition/article.\n"
        "  - Example GOOD: 'Gradient descent minimizes loss by adjusting weights'\n"
        "  - Example GOOD: 'Backward pass computes gradients layer by layer.'\n"
        "  - Example BAD: 'The slide transitions from the high-level goal of training to' (dangles on \"to\")\n"
        "  - Example BAD: 'Organized by defining the objective function first, followed by the' (dangles on \"the\")\n"
        "  - Example BAD: 'Artificial neural networks utilize backpropagation to minimize error rates through' (dangles on \"through\")\n"
        "\nBULLET RULES:\n"
        "  - Each bullet is a FULL EXPLANATORY SENTENCE (14-22 words) that answers BOTH what happens AND why/how.\n"
        "  - Prefer sentences of the form: <subject> <verb> <object>, because/by/when <mechanism or consequence>.\n"
        "  - Each bullet must add NEW information the title alone does not convey.\n"
        "  - Include a concrete mechanism, condition, or effect — not just a category name.\n"
        "  - Example GOOD: 'KNN classifies a query by majority vote over the k closest training samples in feature space.'\n"
        "  - Example GOOD: 'Small k values overfit because single noisy neighbors dominate the decision boundary near the query.'\n"
        "  - Example GOOD: 'Minkowski distance with p=1 yields Manhattan, with p=2 yields Euclidean, unifying common metrics.'\n"
        "  - Example BAD: 'The k-Nearest-Neighbor Rule' (label only, no insight)\n"
        "  - Example BAD: 'Non-Parametric Modeling: No Distribution Assumptions' (colon+noun form, no verb)\n"
        "  - Example BAD: 'Instance-Based Learning Framework' (just a phrase, no mechanism)\n"
        "  - Bullets must read as spoken sentences, not like section headings.\n"
        "\nDIAGRAM NODE LABEL RULES:\n"
        "  - Use HUMAN-READABLE labels (2-4 words), NOT short internal IDs.\n"
        "  - Example GOOD: `h1: Hidden Layer 1 { shape: rectangle }`\n"
        "  - Example BAD:  `h1: h1 { shape: rectangle }`  (label repeats the ID)\n"
        "  - Example BAD:  `vec: vec { shape: cylinder }` (cryptic abbreviation as label)\n"
        "  - Example BAD:  `sum: sum`, `act: act`, `w_old: w_old` (all bad)\n"
        "  - Every node's visible label must be a proper noun phrase, not an acronym/variable name.\n"
        "  - NEVER use `?`, `...`, `TBD`, `<placeholder>` or any single-character placeholder as a node label.\n"
        "  - If a node represents an unknown weight/parameter, label it explicitly (e.g. `Weight w1`, `Hidden Activation`, `Unknown Gradient`) — never just `?`.\n"
        "  - Mathematical expressions in labels MUST be complete and balanced: write `f(x)` not `f(x`, `dy/dx = product` not `dy/dx=∏`. ALWAYS quote labels containing `(`, `)`, `=`, `·`, `×`, `÷`, `≤`, `≥`, `≠` with double quotes — D2 truncates unquoted labels at the first `(`.\n"
        "  - Example GOOD (math): `f1: \"f1(x) = sigmoid\" { shape: rectangle }`\n"
        "  - Example BAD (truncated): `f1: f1(x) { shape: rectangle }` (D2 will render as `f1(x`)\n"
        "\nD2 CODE RULES — each diagram MUST include ALL of these (no exceptions):\n"
        "  1. `direction: right` or `direction: down` as first line\n"
        "  2. MANDATORY: At least 2-3 CONTAINER GROUPS wrapping related nodes:\n"
        "     `group_id: Group Name {\\n  style.fill: \"aliceblue\"\\n  style.stroke-dash: 3\\n  child: Label { shape: oval }\\n  child -> other: action\\n}`\n"
        "     A diagram WITHOUT containers will be rejected.\n"
        "     CRITICAL — CONTAINER LABEL: every container MUST follow `id: Semantic Name { ... }`.\n"
        "     The label after the colon is what the viewer sees — make it a real topic-specific phrase.\n"
        "     BAD: `group1 { ... }` or `grp2 { ... }` → shows 'group1'/'grp2' on screen.\n"
        "     BAD: `inputStage: { ... }` → empty label, shows 'inputStage' on screen.\n"
        "     GOOD: `inputStage: Input Processing { ... }` → shows 'Input Processing'.\n"
        "  3. Each node declared with both a label AND a shape:\n"
        "     `node_id: Node Label { shape: oval }`\n"
        "     CRITICAL — SHAPE ATTRIBUTE: always write `shape: oval`, NEVER a bare `oval` on its own\n"
        "     line — a bare shape name becomes a child node showing that name in a box.\n"
        "     BAD: `bio: Biological Neuron { oval }` → 'oval' renders as a separate child box.\n"
        "     GOOD: `bio: Biological Neuron { shape: oval }` → oval is the shape, no child box.\n"
        "  4. MANDATORY: Each container has a pastel named color fill:\n"
        "     aliceblue, honeydew, lavender, mistyrose, papayawhip, peachpuff, lemonchiffon, mintcream, lavenderblush\n"
        "     Use DIFFERENT colors for different groups in the same diagram.\n"
        "  5. MANDATORY: `style.stroke-dash: 3` on every grouping container\n"
        "  6. MANDATORY: Edge labels on at least 70% of arrows: `source -> dest: action_verb`\n"
        "     e.g., `input -> model: features`, `model -> loss: compute`, `loss -> model: backprop`\n"
        "  7. Use at least 3 different shapes: rectangle, oval, diamond, hexagon, cylinder, cloud\n"
        "  8. Last line: `style.font-size: 32`\n"
        "  9. Each diagram should have 25-40 lines — NOT minimalist\n"
        "  10. The diagram must VISUALLY reflect the slide's specific topic — no generic templates.\n"
        "\nFORBIDDEN: `--` edges, `.south/.north` ports, unclosed braces, markdown fences.\n"
        "\nNODE UNIQUENESS RULES (CRITICAL — prevents duplicate/floating nodes):\n"
        "  - Each NODE ID must be declared EXACTLY ONCE, inside ONE container only.\n"
        "  - NEVER declare the same node both inside a container AND at top level.\n"
        "  - NEVER create helper/floating nodes outside any container.\n"
        "  - Bad (creates duplicates): `mse -> rate` at top level when `mse` is already `error.mse`.\n"
        "  - Good: refer to nested nodes via dotted path: `error.mse -> metrics.rate: computed`.\n"
        "  - All edges between containers must use dotted references: `container1.child -> container2.child: label`.\n"
        "\nLAYOUT HYGIENE RULES (prevents overlapping arrows and off-center nodes):\n"
        "  - Keep each diagram roughly balanced: containers should have similar numbers of nodes (2-4 each), not one big and one tiny.\n"
        "  - Do NOT create standalone single-node containers; a container must have at least 2 children.\n"
        "  - Prefer short local edges inside a container over long cross-diagram edges that span multiple groups.\n"
        "  - Use at most 2 cross-container edges per diagram, and put their labels on short segments.\n"
        "  - Keep edge labels to 1-2 words so routing does not push them onto other nodes.\n"
        "  - Avoid edges that cross each other; redesign the flow rather than relying on the layout engine.\n"
        "  - Every leaf node must have a distinct visual role — no orphan circles floating without an explicit incoming or outgoing edge.\n"
        "\nNATURAL LANGUAGE RULES (so it reads like a human teacher wrote it):\n"
        "  - Bullets: start with a clear subject, avoid stiff academic jargon walls.\n"
        "  - Speech: conversational but precise; short sentences that a professor would actually say out loud.\n"
        "  - Diagram labels: use real English phrases (Hidden Layer 1, Loss Function) not abbreviations (h1, loss_fn).\n"
        "  - Edge labels: 1-2 words describing the action (computes, feeds, updates, normalizes).\n"
        "  - Each slide must sound coherent and distinct — do not paraphrase the same idea across slides.\n"
        "\nEXAMPLE of rich D2 (copy this style — containers, colors, labels, shapes):\n"
        "  ```\n"
        "  direction: right\n"
        "  input_stage: Input Processing {\n"
        "    style.fill: \"aliceblue\"\n"
        "    style.stroke-dash: 3\n"
        "    raw: Raw Data { shape: cylinder }\n"
        "    clean: Preprocess { shape: rectangle }\n"
        "    raw -> clean: normalize\n"
        "  }\n"
        "  model: Neural Network {\n"
        "    style.fill: \"honeydew\"\n"
        "    style.stroke-dash: 3\n"
        "    h1: Hidden 1 { shape: rectangle }\n"
        "    h2: Hidden 2 { shape: rectangle }\n"
        "    act: Activation { shape: diamond }\n"
        "    h1 -> act: output\n"
        "    act -> h2: input\n"
        "  }\n"
        "  eval: Evaluation {\n"
        "    style.fill: \"mistyrose\"\n"
        "    pred: Prediction { shape: oval }\n"
        "    loss: Loss { shape: hexagon }\n"
        "    pred -> loss: compute\n"
        "  }\n"
        "  input_stage.clean -> model.h1: features\n"
        "  model.h2 -> eval.pred: predict\n"
        "  eval.loss -> model.h1: backprop\n"
        "  style.font-size: 32\n"
        "  ```\n"
        "\nCOPY THIS STYLE — make your diagrams equally rich with:\n"
        " * 3+ container groups with different pastel colors\n"
        " * Explicit shape per node\n"
        " * Labeled edges with 1-3 word action verbs\n"
        " * Cross-group arrows to show relationships\n"
        "\nReturn a single valid JSON object. The `slides` array must have EXACTLY "
        f"{req.slides_count} items. No markdown fences outside diagram_code.\n"
    )



# PROMPT 1 — Global plan (runs once for the entire deck)


def global_plan_prompt(req: PresentationRequest, outline: list[SlideOutline], rag_summary: str) -> str:
    """Build a prompt that creates a global plan with unique ideas and diagram themes per slide.

    This prevents diagram repetition by planning all slides at once.
    """
    slides_block = ""
    for s in outline:
        slides_block += f"  Slide {s.slide_number} ({s.type}): \"{s.title}\" — goal: {s.goal}\n"

    return (
        "You are planning an academic presentation. Create a GLOBAL PLAN that assigns "
        "a unique main idea and a unique diagram theme to each slide.\n"
        "\n"
        f"Topic: {req.topic}\n"
        f"Style: {req.style}\n"
        f"Total slides: {req.slides_count}\n"
        "\nSlide outline:\n"
        f"{slides_block}"
        "\nRAG knowledge summary:\n"
        f"{rag_summary}\n"
        "\nReturn JSON with key `plan` — an array with one object per slide.\n"
        "Each object must have:\n"
        "- slide_number (int)\n"
        "- slide_role: one of `intro` | `build_context` | `mechanism` | `example` | `connection` | `conclusion`.\n"
        "  Defines what THIS slide does for the listener. The deck must include exactly one\n"
        "  `intro` (slide 1), exactly one `conclusion` (last slide), and a mix of the rest in\n"
        "  the middle so each slide plays a distinct role in the narrative.\n"
        "- main_idea (1 sentence — the ONE unique concept this slide teaches)\n"
        "- diagram_theme (1 sentence — what the diagram should illustrate, DIFFERENT from all other slides)\n"
        "- progression (how this slide builds on the previous one)\n"
        "\nRules:\n"
        "- Slide 1 (`intro`): overview/introduction diagram — show the big picture.\n"
        "- Middle slides: alternate between `build_context` (motivation, prerequisites),\n"
        "  `mechanism` (how the algorithm works step by step), `example` (concrete instance,\n"
        "  formula, numerical demo), and `connection` (link to other concepts or applications).\n"
        "  Each must cover a DIFFERENT aspect. No two slides should have similar diagrams.\n"
        "- Last slide (`conclusion`): summary diagram connecting all key concepts.\n"
        "- Progression must flow logically: general → specific → applied → summary.\n"
        "- Each diagram_theme must be visually distinct (different structure: flow, hierarchy, cycle, comparison, etc.)\n"
        "- main_idea <= 20 words; diagram_theme <= 25 words; progression <= 15 words.\n"
        "- Return a single valid JSON object, no markdown, no prose outside JSON.\n"
    )



# PROMPT 2 — Combined slide detail + diagram (1 call per slide)
#            Produces bullets, speech, diagram_description, diagram_code
#            in a single Gemini call to stay under free-tier rate limits.


def slide_combined_prompt(
    req: PresentationRequest,
    outline: SlideOutline,
    *,
    global_plan_text: str,
    rag_entries_text: str,
    min_bullets: int,
    max_bullets: int,
) -> str:
    """Single-call prompt: detail + diagram concept + D2 code, with global plan in context."""
    bullets_rule = f"{max(1, int(min_bullets))} to {max(1, int(max_bullets))}"
    return (
        "You are detailing ONE slide of an academic presentation. Produce text content AND a D2 diagram.\n"
        "Follow the GLOBAL PLAN below — do NOT repeat ideas from other slides.\n"
        "\n"
        f"Topic: {req.topic}\n"
        f"Style: {req.style}\n"
        f"Slide number: {outline.slide_number}\n"
        f"Slide type: {outline.type}\n"
        f"Title draft: {outline.title}\n"
        f"Goal: {outline.goal}\n"
        "\nGLOBAL PLAN (all slides):\n"
        f"{global_plan_text}\n"
        "\nRAG entries for this slide:\n"
        f"{rag_entries_text}\n"
        "\nReturn JSON with keys:\n"
        "- teaching_scenario (short paragraph, <= 55 words)\n"
        "- structure_plan (how info is organized, <= 35 words)\n"
        f"- bullet_plan (array of {bullets_rule} objects; each has `bullet` <= 12 words and `explanation` <= 20 words)\n"
        "- diagram_description (structured text, see format below)\n"
        "- diagram_code (valid D2 code, see rules below)\n"
        "- speech_plan (speaker script, <= 110 words, must include RAG facts)\n"
        "- layout (object) — YOU pick the visual composition per slide. Required fields:\n"
        "    `kind`: `text-dominant` | `balanced` | `diagram-dominant` | `diagram-only` | `diagram-above-bullets`\n"
        "    `text_ratio`: int 25-65, % slide width for text column (lower = bigger diagram)\n"
        "    `title_size_em`: float 1.0-2.0 (smaller for long titles)\n"
        "    `bullet_size_em`: float 0.8-1.3 (bigger for few/short bullets)\n"
        "    `bullet_line_height`: float 1.2-1.8\n"
        "    `diagram_max_height_pct`: int 50-100\n"
        "    `diagram_max_width_pct`: int 50-100\n"
        "    `accent_color`: hex, default `#005ab4`\n"
        "    `background`: `white` or soft hex\n"
        "    `notes`: <=20 words why\n"
        "\nRules for bullets/speech:\n"
        "- Ground every idea in RAG entries. Reuse at least 2 technical terms.\n"
        "- Professor tone for undergraduate students.\n"
        "- Bullets = one teachable idea each, concrete and specific.\n"
        "- This slide must be DIFFERENT from all other slides in the plan.\n"
        "\nRules for diagram_description:\n"
        "  Format: `nodes: NodeA (shape, Label), NodeB (shape, Label), ... | "
        "arrows: NodeA -> NodeB, ... | direction: right or down | notes: hints`\n"
        "  - 6-10 nodes with short labels (2-4 words).\n"
        "  - Allowed shapes: rectangle, oval, diamond, hexagon, cylinder, cloud.\n"
        "  - Labels must be semantic concepts, not shape names or placeholders.\n"
        "    BAD labels: `rectangle`, `diamond`, `cloud`, `node1`, `group2`.\n"
        "    GOOD labels: `Loss Function`, `Decision Boundary`, `Feature Store`.\n"
        "  - 5-12 arrows, at least 3 different shapes.\n"
        "  - Include at least one branching/merging path and one skip/feedback connection.\n"
        "  - Match the diagram_theme from the global plan — be UNIQUE.\n"
        "\nRules for diagram_code (D2 syntax):\n"
        "  - First line: `direction: right` or `direction: down`\n"
        "  - Node: `id: Label { shape: oval }`\n"
        "  - Container (only for pipeline/hierarchy): `grp: Group Name { child: X }` — MUST have human name after colon\n"
        "  - Style: `style.fill: \"aliceblue\"` (aliceblue, honeydew, lavender, mistyrose, papayawhip, peachpuff)\n"
        "  - Edge: `a -> b: label`\n"
        "  - Last line: `style.font-size: 32`\n"
        "  - Match diagram structure to diagram_theme:\n"
        "    cycle/loop → flat ring of nodes, NO containers\n"
        "    comparison → flat parallel nodes, NO containers\n"
        "    flow/pipeline → 2-3 named stage containers\n"
        "    all other conceptual/mechanism slides → prefer 2-3 named containers unless a flat layout is clearly better\n"
        "  - Every `{` must have matching `}`. NO `--` edges. Max 45 lines.\n"
        "\nReturn a single valid JSON object. No markdown fences outside the diagram_code value.\n"
    )



# PROMPT 2.1 — Slide detail + diagram theme (per slide, with global plan)


_WRITING_STYLE_GUIDELINES = {
    "formal-academic": (
        "Write in a formal academic register. Prefer precise noun phrases and "
        "passive constructions where appropriate. Sentences should read like a "
        "textbook: definitional, measured, authoritative. Avoid first/second "
        "person and contractions."
    ),
    "conversational-teaching": (
        "Write as a professor explaining out loud to a class. Active voice, "
        "direct, occasionally addressing the reader with `you`. Short declarative "
        "sentences that feel spoken rather than written."
    ),
    "intuition-first": (
        "Lead each bullet with a plain-language intuition or analogy, then name "
        "the technical term. Prefer verbs of motion / change / comparison "
        "(`pushes`, `shrinks`, `mirrors`). Avoid starting with jargon."
    ),
    "historical-progression": (
        "Frame ideas as a historical progression — what came first, what "
        "replaced it, what the current best practice is. Use temporal markers "
        "(`originally`, `then`, `modern`, `recent`). Each bullet should feel "
        "like a step forward in time."
    ),
    "engineer-practical": (
        "Write like an engineer documenting a system. Concrete nouns, "
        "implementation verbs (`caches`, `stores`, `pipelines`, `batches`). "
        "Mention data flow, memory, compute cost when relevant. Avoid "
        "philosophical framing."
    ),
    "math-rigorous": (
        "Write with precise mathematical framing. Use notation where it "
        "clarifies (∂, ∇, Σ, θ, α via Unicode — never LaTeX backslashes). "
        "State relationships as equalities or orderings where possible."
    ),
}


def _writing_style_block(style: str) -> str:
    """Return a WRITING STYLE directive for the given style, or empty string.

    NOTE: this is *lexical* style only — vocabulary, sentence rhythm, and
    framing. It has nothing to do with TTS / voiceover audio.
    """
    guideline = _WRITING_STYLE_GUIDELINES.get((style or "").strip().lower())
    if not guideline:
        return ""
    return (
        "\nWRITING STYLE for this run — apply it to every bullet, explanation,\n"
        f"and speech field: {guideline}\n"
        "This run must sound LEXICALLY different from other runs on the same\n"
        "topic. Vary word choice, sentence rhythm, and framing accordingly.\n"
    )


def slide_detail_prompt(
    req: PresentationRequest,
    outline: SlideOutline,
    *,
    global_plan_text: str,
    rag_entries_text: str,
    min_bullets: int,
    max_bullets: int,
) -> str:
    """Build a prompt for slide-level detail, given the global plan as context.

    This ensures each slide knows what all other slides cover, preventing repetition.
    """
    bullets_rule = f"{max(1, int(min_bullets))} to {max(1, int(max_bullets))}"
    style_block = _writing_style_block(os.environ.get("WRITING_STYLE", ""))
    audience = (os.environ.get("AUDIENCE", "") or "undergraduate students").strip()
    return (
        "You are detailing ONE slide of an academic presentation. After drafting,\n"
        "act as a RUTHLESS EDITOR on your own output: remove fluff, sharpen each\n"
        "bullet, ensure exactly ONE core idea per bullet, prefer concrete\n"
        "mechanism over abstract category names.\n"
        "You have the GLOBAL PLAN below — follow it strictly. "
        "Do NOT repeat ideas from other slides.\n"
        "\n"
        f"Topic: {req.topic}\n"
        f"Style: {req.style}\n"
        f"Audience: {audience}\n"
        f"Slide number: {outline.slide_number}\n"
        f"Slide type: {outline.type}\n"
        f"Title draft: {outline.title}\n"
        f"Goal: {outline.goal}\n"
        f"{style_block}"
        "\nGLOBAL PLAN (for all slides — shows what each slide covers):\n"
        f"{global_plan_text}\n"
        "\nRAG entries for this slide:\n"
        f"{rag_entries_text}\n"
        "\nReturn JSON with keys:\n"
        "- teaching_scenario (short paragraph, professor perspective, <= 55 words)\n"
        "- structure_plan (how information is organized on this slide, <= 35 words)\n"
        f"- bullet_plan (array of {bullets_rule} objects; each has `bullet` 7-12 words as a COMPLETE, SELF-CONTAINED sentence and `explanation` <= 20 words)\n"
        "- diagram_theme (copy from global plan — what this diagram illustrates)\n"
        "- speech_plan (speaker script, <= 110 words, must include RAG facts)\n"
        "- layout (object) — YOU only pick stylistic flavor; sizes and column split are computed\n"
        "  automatically by the renderer from the diagram's PNG aspect ratio, so DO NOT try to\n"
        "  resize text vs image — just set flavor.\n"
        "  Required fields:\n"
        "    `kind` — one of `text-dominant` | `balanced` | `diagram-dominant` | `diagram-only`\n"
        "    `title_size_em` — float 1.2-1.8, header font size (smaller for long titles)\n"
        "    `accent_color` — hex color for the line under the title, default `#005ab4`\n"
        "    `background` — `white` or a soft hex like `#fff8f0`, `#f3f8ff`\n"
        "    `notes` — <=20 words explaining WHY you picked these values\n"
        "  Guidelines: long title (>8 words) -> title_size_em 1.2-1.3; short punchy title -> 1.6-1.8.\n"
        "\nRules:\n"
        "- Ground every idea in RAG entries. Reuse at least 2 technical terms from RAG.\n"
        "- bullet_plan: each `bullet` MUST be a complete sentence (subject + verb) ending in a period,\n"
        "  AND must read as a finished thought on its own — NEVER end on `and`, `the`, `of`, `a`,\n"
        "  any trailing clause fragment, OR a trailing operator / inequality / unit (`≈`, `<`,\n"
        "  `>`, `=`, `+`, `-`, `×`, `·`, `≤`, `≥`, `→`). If you write `c ≈`, finish the value\n"
        "  (`c ≈ 2`); never leave the reader hanging on the symbol. Keep it tight: 7-12 words total.\n"
        "  GOOD: `Backpropagation computes gradients via the chain rule.` (8 words, complete)\n"
        "  GOOD: `Gradient descent updates weights to minimize the loss.` (9 words, complete)\n"
        "  GOOD: `Each layer multiplies its incoming gradient by ∂z/∂w to produce the parameter gradient.`\n"
        "  BAD:  `The algorithm iteratively adjusts weights through a cycle of forward pass, loss` (trails off)\n"
        "  BAD:  `Backpropagation: chain rule in action` (no verb, no period)\n"
        "  BAD:  `Total operations satisfy T_backward ≤ c · T_forward for constant c ≈` (trailing `≈` with no value)\n"
        "- TEACHING focus: each bullet should answer ONE of {what is it, how does it work,\n"
        "  what is it good for}. Mix the three across the 2-3 bullets per slide. Avoid math-paper\n"
        "  framing where every bullet is a formula — formulas belong in the diagram, the bullet\n"
        "  text should explain the formula in words a student can read aloud.\n"
        "- This slide's content must be DIFFERENT from all other slides in the plan.\n"
        f"- Adapt tone, vocabulary, and depth to the audience: {audience}.\n"
        "  Avoid jargon the audience would not know; when a domain term is unavoidable,\n"
        "  let the bullet itself unpack what it means.\n"
        "\nRUTHLESS EDITOR PASS — apply BEFORE returning JSON:\n"
        "- Strip filler words (`actually`, `simply`, `essentially`, `in order to`, hedge clauses).\n"
        "- Replace abstract category names with the concrete mechanism (`computes the gradient`\n"
        "  beats `enables optimization`).\n"
        "- Each bullet must add NEW information the title alone does not convey.\n"
        "- If two bullets restate the same idea differently, drop one and use the saved\n"
        "  slot for a fresh angle (an example, a constraint, a consequence).\n"
        "- Prefer concrete numbers / formulas / units when they make the point shorter.\n"
        "- Avoid stacking adjectives (`a very large complex deep network` -> `a deep network`).\n"
        "- Return a single valid JSON object, no markdown, no prose outside JSON.\n"
    )



# PROMPT 2.2 — Diagram concept (per slide, with slide detail)


def diagram_concept_prompt(
    slide_number: int,
    slide_title: str,
    diagram_theme: str,
    bullet_plan_text: str,
    rag_entries_text: str,
    diagram_signature: str = "",
) -> str:
    """Build a prompt that designs the diagram at conceptual level.

    Output: nodes, arrows, shapes, direction — NOT code.

    `diagram_signature` (optional) is the per-slide structural family from
    the global plan (`flow`, `hierarchy`, `cycle`, `comparison`, `stack`,
    `matrix`, `tree`, `state-machine`, `pipeline`). When provided it pins
    the model to that structural family instead of letting it default to
    a generic left-to-right flow.
    """
    sig_line = ""
    if diagram_signature:
        sig_line = (
            f"Diagram signature (from global plan): {diagram_signature}.\n"
            "  YOU MUST realize this exact structural family — do NOT default to a generic\n"
            "  left-to-right `flow`. Map signature -> structure:\n"
            "  - `flow`: linear or branching arrows, left-to-right.\n"
            "  - `hierarchy`: tree of containers, root at top, children below.\n"
            "  - `cycle`: closed loop of 4-6 nodes connecting back to start.\n"
            "  - `comparison`: two parallel columns side-by-side with the same shape repeated.\n"
            "  - `stack`: vertical stack of containers, one per layer.\n"
            "  - `matrix`: grid of rows x columns of small uniform nodes.\n"
            "  - `tree`: parent expanding into multiple children, recursive.\n"
            "  - `state-machine`: nodes with self-loops and bidirectional transitions.\n"
            "  - `pipeline`: stages with named inputs/outputs between each.\n"
        )
    return (
        "You are designing an educational diagram for a presentation slide.\n"
        "Describe the diagram at CONCEPTUAL level — list nodes, arrows, shapes, direction.\n"
        "Do NOT write any code. Just describe the structure.\n"
        "\n"
        f"Slide {slide_number}: \"{slide_title}\"\n"
        f"Diagram theme: {diagram_theme}\n"
        f"{sig_line}"
        f"\nBullet points on this slide:\n{bullet_plan_text}\n"
        f"\nRAG context:\n{rag_entries_text}\n"
        "\nReturn JSON with key `diagram_description` using this exact format:\n"
        '  "nodes: NodeA (shape, Label), NodeB (shape, Label), ... | '
        "arrows: NodeA -> NodeB, NodeB -> NodeC, ... | "
        'direction: right or down | notes: layout hints"\n'
        "\n"
        "Requirements:\n"
        "- 5 to 10 nodes, each with a SHORT label (1-3 words, ≤ 18 characters)\n"
        "  and a shape. Longer labels overflow the rendered box.\n"
        "  Allowed shapes: rectangle, oval, diamond, hexagon, cylinder, cloud.\n"
        "- Labels must be semantic concept names, never raw shape names or placeholder IDs.\n"
        "  BAD: `rectangle`, `diamond`, `cloud`, `node1`, `grp1`.\n"
        "  GOOD: `Training Loss`, `Hidden Layer`, `Feature Store`.\n"
        "- 4 to 12 arrows as Source -> Destination.\n"
        "- At least 3 different shapes.\n"
        "- Prefer a visually rich but readable topology: one clear main path plus\n"
        "  one secondary branch, skip, feedback, or merge relation.\n"
        "- At least one branching/merging path (not just linear A->B->C).\n"
        "- At least one feedback or skip connection.\n"
        "- Structure must match the diagram_theme:\n"
        "  cycle/loop/iterative → flat ring of nodes, mention 'no containers'\n"
        "  comparison/vs/tradeoff → flat parallel columns, mention 'no containers'\n"
        "  flow/pipeline/process → mention stage names for 2-3 containers\n"
        "  hierarchy/tree/taxonomy → parent node fanning out to children\n"
        "- Richness target by structure:\n"
        "  cycle/comparison/state-machine → usually 5-7 nodes, keep it clean and uncluttered\n"
        "  flow/pipeline/hierarchy/tree/stack → prefer 7-10 nodes with 2-3 grouped stages or branches when natural\n"
        "  if the slide mentions phases, modules, data flow, training steps, or components, explicitly represent them as separate nodes/groups\n"
        "- Default preference: use 2-3 named groups/containers for mechanism, pipeline, architecture,\n"
        "  hierarchy, and multi-stage explanation slides. Use flat layouts mainly for cycle,\n"
        "  comparison, or simple fan-out diagrams.\n"
        "- Where the content allows it, prefer a richer diagram over a minimal one: include\n"
        "  intermediate transformations, decision points, storage nodes, or feedback paths instead of collapsing everything into 4-5 nodes.\n"
        "- If the topic has a formula, include ONE equation node with the\n"
        "  formula as Unicode symbols inside the label (never LaTeX backslashes).\n"
        "- Every node MUST appear in at least one arrow — no orphan boxes.\n"
        "\n"
        "Example for a cycle diagram:\n"
        '  "nodes: Train (hexagon, Train Model), Validate (diamond, Validate), '
        "Deploy (oval, Deploy), Monitor (cylinder, Monitor), Retrain (rectangle, Retrain) | "
        "arrows: Train -> Validate, Validate -> Deploy: if good, Validate -> Train: if poor, "
        "Deploy -> Monitor, Monitor -> Retrain, Retrain -> Train | "
        'direction: right | notes: closed loop, no containers"\n'
        "\n"
        "Example for a neural network (flow with stages):\n"
        '  "nodes: Input (oval, Input Data), H1 (rectangle, Hidden Layer 1), '
        "H2 (rectangle, Hidden Layer 2), Output (diamond, Predictions), "
        "Loss (hexagon, Loss Function) | "
        "arrows: Input -> H1, H1 -> H2, H2 -> Output, Output -> Loss, Input -> H2 | "
        'direction: right | notes: skip connection from Input to H2"\n'
        "\n"
        "- diagram_description <= 210 words.\n"
        "- Return a single valid JSON object, no markdown, no prose outside JSON.\n"
    )


def diagram_concept_prompt_v2(
    slide_number: int,
    slide_title: str,
    diagram_theme: str,
    bullet_plan_text: str,
    rag_entries_text: str,
    prolog_facts: list[str],
    prolog_rules: list[str],
    diagram_signature: str = "",
) -> str:
    """V2 variant of diagram_concept_prompt — includes a Prolog KB section.

    The KB (facts + rules) was extracted from the slide bullets either via
    spaCy (V2_spacy, no LLM) or via LLM (V2_llm).  Giving the diagram designer
    an explicit relational structure reduces generic flowcharts and grounds
    nodes/arrows in the actual content relationships.

    `diagram_signature` has the same semantics as in diagram_concept_prompt().
    """
    sig_line = ""
    if diagram_signature:
        sig_line = (
            f"Diagram signature (from global plan): {diagram_signature}.\n"
            "  YOU MUST realize this exact structural family — do NOT default to a generic\n"
            "  left-to-right `flow`. Map signature -> structure:\n"
            "  - `flow`: linear or branching arrows, left-to-right.\n"
            "  - `hierarchy`: tree of containers, root at top, children below.\n"
            "  - `cycle`: closed loop of 4-6 nodes connecting back to start.\n"
            "  - `comparison`: two parallel columns side-by-side with the same shape repeated.\n"
            "  - `stack`: vertical stack of containers, one per layer.\n"
            "  - `matrix`: grid of rows x columns of small uniform nodes.\n"
            "  - `tree`: parent expanding into multiple children, recursive.\n"
            "  - `state-machine`: nodes with self-loops and bidirectional transitions.\n"
            "  - `pipeline`: stages with named inputs/outputs between each.\n"
        )

    kb_lines: list[str] = []
    if prolog_facts:
        kb_lines.append("% Prolog facts (subject-relation-object triples from slide text):")
        kb_lines.extend(prolog_facts)
    if prolog_rules:
        kb_lines.append("% Prolog rules (conditional relationships):")
        kb_lines.extend(prolog_rules)
    kb_block = (
        "\nKnowledge base (Prolog, extracted from slide text):\n"
        "  Use these facts and rules to identify the KEY ENTITIES (→ diagram nodes)\n"
        "  and RELATIONSHIPS (→ arrows). Predicates name edge labels, arguments name nodes.\n"
        "  Rules suggest conditional or derived edges.\n"
        + "\n".join(kb_lines) + "\n"
        if kb_lines else ""
    )

    return (
        "You are designing an educational diagram for a presentation slide.\n"
        "Describe the diagram at CONCEPTUAL level — list nodes, arrows, shapes, direction.\n"
        "Do NOT write any code. Just describe the structure.\n"
        "\n"
        f"Slide {slide_number}: \"{slide_title}\"\n"
        f"Diagram theme: {diagram_theme}\n"
        f"{sig_line}"
        f"\nBullet points on this slide:\n{bullet_plan_text}\n"
        f"\nRAG context:\n{rag_entries_text}\n"
        f"{kb_block}"
        "\nReturn JSON with key `diagram_description` using this exact format:\n"
        '  "nodes: NodeA (shape, Label), NodeB (shape, Label), ... | '
        "arrows: NodeA -> NodeB, NodeB -> NodeC, ... | "
        'direction: right or down | notes: layout hints"\n'
        "\n"
        "Requirements:\n"
        "- 5 to 10 nodes, each with a SHORT label (1-3 words, ≤ 18 characters)\n"
        "  and a shape. Longer labels overflow the rendered box.\n"
        "  Allowed shapes: rectangle, oval, diamond, hexagon, cylinder, cloud.\n"
        "- Labels must be semantic concept names, never raw shape names or placeholder IDs.\n"
        "  BAD: `rectangle`, `diamond`, `cloud`, `node1`, `grp1`.\n"
        "  GOOD: `Training Loss`, `Hidden Layer`, `Feature Store`.\n"
        "- 4 to 12 arrows as Source -> Destination.\n"
        "- At least 3 different shapes.\n"
        "- Prefer a visually rich but readable topology: one clear main path plus\n"
        "  one secondary branch, skip, feedback, or merge relation.\n"
        "- At least one branching/merging path (not just linear A->B->C).\n"
        "- At least one feedback or skip connection.\n"
        "- Structure must match the diagram_theme:\n"
        "  cycle/loop/iterative → flat ring of nodes, mention 'no containers'\n"
        "  comparison/vs/tradeoff → flat parallel columns, mention 'no containers'\n"
        "  flow/pipeline/process → mention stage names for 2-3 containers\n"
        "  hierarchy/tree/taxonomy → parent node fanning out to children\n"
        "- Richness target by structure:\n"
        "  cycle/comparison/state-machine → usually 5-7 nodes, keep it clean and uncluttered\n"
        "  flow/pipeline/hierarchy/tree/stack → prefer 7-10 nodes with 2-3 grouped stages or branches when natural\n"
        "  if the slide mentions phases, modules, data flow, training steps, or components, explicitly represent them as separate nodes/groups\n"
        "- Default preference: use 2-3 named groups/containers for mechanism, pipeline, architecture,\n"
        "  hierarchy, and multi-stage explanation slides. Use flat layouts mainly for cycle,\n"
        "  comparison, or simple fan-out diagrams.\n"
        "- Where the content allows it, prefer a richer diagram over a minimal one: include\n"
        "  intermediate transformations, decision points, storage nodes, or feedback paths instead of collapsing everything into 4-5 nodes.\n"
        "- If the topic has a formula, include ONE equation node with the\n"
        "  formula as Unicode symbols inside the label (never LaTeX backslashes).\n"
        "- Every node MUST appear in at least one arrow — no orphan boxes.\n"
        "- If a Prolog knowledge base was provided above, map its predicates to edge\n"
        "  labels and its arguments to node names wherever they fit the theme.\n"
        "\n"
        "Example for a cycle diagram:\n"
        '  "nodes: Train (hexagon, Train Model), Validate (diamond, Validate), '
        "Deploy (oval, Deploy), Monitor (cylinder, Monitor), Retrain (rectangle, Retrain) | "
        "arrows: Train -> Validate, Validate -> Deploy: if good, Validate -> Train: if poor, "
        "Deploy -> Monitor, Monitor -> Retrain, Retrain -> Train | "
        'direction: right | notes: closed loop, no containers"\n'
        "\n"
        "Example for a neural network (flow with stages):\n"
        '  "nodes: Input (oval, Input Data), H1 (rectangle, Hidden Layer 1), '
        "H2 (rectangle, Hidden Layer 2), Output (diamond, Predictions), "
        "Loss (hexagon, Loss Function) | "
        "arrows: Input -> H1, H1 -> H2, H2 -> Output, Output -> Loss, Input -> H2 | "
        'direction: right | notes: skip connection from Input to H2"\n'
        "\n"
        "- diagram_description <= 210 words.\n"
        "- Return a single valid JSON object, no markdown, no prose outside JSON.\n"
    )



# Legacy prompts (kept for backward compatibility)


def final_layout_prompt(
    *,
    slide_w: int,
    slide_h: int,
    image_w: int,
    image_h: int,
    title: str,
    bullets: list[str],
    draft_layout: dict,
) -> str:
    """Prompt 3: pick the FINAL layout for a slide given concrete dimensions.

    Called AFTER the diagram PNG is rendered, so the model has real pixel
    sizes (slide canvas + image) plus the actual bullet text — instead of
    guessing layout from a D2 source. Returns a small JSON object that
    overrides the draft layout from the scenario plan.
    """
    bullet_lines = []
    for i, b in enumerate(bullets, 1):
        bullet_lines.append(f"  {i}. \"{b}\" ({len(b.split())} words, {len(b)} chars)")
    bullets_text = "\n".join(bullet_lines) if bullet_lines else "  (no bullets)"

    image_ratio = image_w / max(1, image_h)
    if image_ratio >= 1.4:
        shape_hint = "WIDE (longer horizontally than vertically)"
    elif image_ratio <= 0.85:
        shape_hint = "TALL (longer vertically than horizontally)"
    else:
        shape_hint = "ROUGHLY SQUARE"

    draft_pos = str(draft_layout.get("image_position", "")).strip().lower() or "(unset)"
    draft_text_ratio = draft_layout.get("text_ratio", "(unset)")

    return (
        "You are a SLIDE LAYOUT DESIGNER. Think like a human typesetter:\n"
        "FIRST decide where the diagram fits best so it stays VISIBLE and READABLE,\n"
        "THEN flow the bullet text into the remaining space.\n"
        "\n"
        "## Slide canvas\n"
        f"Width: {slide_w}px, Height: {slide_h}px (16:9). Title row takes ~80-110px; "
        "the rest is the content area for the diagram and bullets.\n"
        "\n"
        "## Diagram (already rendered)\n"
        f"Width: {image_w}px, Height: {image_h}px, aspect ratio: {image_ratio:.2f} -> {shape_hint}.\n"
        "\n"
        "## Slide title\n"
        f"\"{title}\"\n"
        "\n"
        "## Bullets (will appear on the slide as text)\n"
        f"{bullets_text}\n"
        "\n"
        "## Draft layout from earlier prompt (you MAY override)\n"
        f"image_position: {draft_pos}, text_ratio: {draft_text_ratio}\n"
        "\n"
        "## Design principles — apply judgment, not rigid rules\n"
        "A. Make the diagram VISIBLE first. Pick the position that gives it the widest\n"
        "   readable footprint without cramping the text.\n"
        "   - Very wide flow (ratio >= 2.0): MUST be `bottom`. Side columns would crush it.\n"
        "   - Moderately wide (1.0 <= ratio < 2.0): PREFER `right` (or `left` for variety) —\n"
        "     stacking these wastes vertical space and pushes the image into the bullets.\n"
        "     Only pick `bottom` if the diagram has long horizontal labels that need width.\n"
        "   - Roughly square / tall (ratio < 1.0): `right` (or `left`). Never `top`/`bottom`.\n"
        "   - Aim for VARIETY across the deck: don't put every slide in the same position.\n"
        "B. SCALE the diagram to fill its area. Set `image_size_pct` (50-100) — how much of\n"
        "   the available image space it should occupy. Generous defaults:\n"
        "   - Wide diagrams in `bottom`/`top`: 95-100 (fill width)\n"
        "   - Tall diagrams on the side: 90-100 (fill height)\n"
        "   - Sparse diagrams (few nodes, lots of whitespace in the PNG): 100 (let it grow)\n"
        "C. SIZE the text to the volume of bullets. Use `bullet_size_em` 0.85-1.15:\n"
        "   <= 120 chars total -> 1.10; 120-200 -> 1.00; 200-300 -> 0.92; > 300 -> 0.85.\n"
        "   Stacked layouts subtract 0.05 (less vertical space for text).\n"
        "D. Title sizing: <= 6 words -> 1.6em; 7-10 words -> 1.4em; > 10 words -> 1.25em.\n"
        "E. `bullet_line_height` 1.25-1.45 (denser bullets -> tighter).\n"
        "F. `text_ratio` only matters for `right`/`left`: 38-48 for square diagrams (image gets\n"
        "   more), 50-58 for tall diagrams (text gets more horizontal room).\n"
        "\n"
        "## Output\n"
        "Return ONLY a JSON object with these exact keys, no markdown, no prose:\n"
        "{\n"
        '  "image_position": "right" | "left" | "top" | "bottom",\n'
        '  "image_size_pct": int 50-100,\n'
        '  "text_ratio": int 30-60,\n'
        '  "bullet_size_em": float 0.80-1.15,\n'
        '  "bullet_line_height": float 1.20-1.50,\n'
        '  "title_size_em": float 1.0-1.8,\n'
        '  "reason": "<= 25 words explaining the choice"\n'
        "}\n"
    )


def outline_prompt(req: PresentationRequest) -> str:
    """Build the strict JSON prompt used for outline generation.

    Two modes:
      * AUTO   — `req.slides_count == 0`: the model picks the number of
                 slides from [6, 12] based on topic density.
      * TARGET — `req.slides_count > 0`: model may still deviate by -2/+3,
                 so the target is a soft guideline, not a hard cap.
    """
    auto = int(req.slides_count) <= 0
    if auto:
        lo, hi = 5, 15
        count_line = (
            f"You choose the slide count ({lo}-{hi}) based on how much teachable "
            "material the topic really contains — do NOT pad or repeat ideas. "
            "A narrow topic may need 5 slides; a rich topic may need 15."
        )
        length_rule = f"- the final array must have length between {lo} and {hi}, inclusive.\n"
    else:
        target = max(3, int(req.slides_count))
        lo = max(3, target - 2)
        hi = min(20, target + 3)
        count_line = (
            f"Target slide count: {target} (you may return between {lo} and {hi} "
            "slides based on how much teachable material the topic really contains — "
            "split dense topics into more slides, merge thin topics into fewer)."
        )
        length_rule = f"- the final array must have length between {lo} and {hi}, inclusive.\n"

    style_block = _writing_style_block(os.environ.get("WRITING_STYLE", ""))
    return (
        "Generate a professional academic presentation outline in English.\n"
        f"Topic: {req.topic}\n"
        f"{count_line}\n"
        f"{style_block}"
        "Return JSON only with key `slides`.\n"
        "Each item must contain:\n"
        "- slide_number (int, 1-based, sequential)\n"
        "- type (title|content|section|conclusion)\n"
        "- title\n"
        "- goal (one sentence explaining what this slide teaches)\n"
        "Rules:\n"
        "- slide 1 must be type `title`.\n"
        "- last slide must be type `conclusion`.\n"
        "- middle slides should usually be `content`.\n"
        "- keep titles specific and technical, not generic.\n"
        "- every content slide MUST teach ONE distinct idea — do NOT pad with\n"
        "  redundant slides just to hit any number. Prefer fewer, stronger slides\n"
        "  over repetition.\n"
        f"{length_rule}"
    )


def _voiceover_rule(
    *,
    req: PresentationRequest,
    is_content: bool,
    target_total_seconds: int,
    voiceover_profile: str,
) -> str:
    """Compute target narration length guidance from profile and deck duration."""
    slides_count = max(1, int(req.slides_count))
    target_seconds = max(90, int(target_total_seconds))
    total_words = int(target_seconds * 2.35)
    avg_words = max(18, total_words // slides_count)

    profile = (voiceover_profile or "short").strip().lower()
    if profile not in {"short", "standard", "long"}:
        profile = "short"

    if profile == "short":
        if is_content:
            low, high = max(24, avg_words - 12), max(40, avg_words + 8)
        else:
            low, high = max(16, int(avg_words * 0.55)), max(28, int(avg_words * 0.75))
    elif profile == "long":
        if is_content:
            low, high = max(60, avg_words + 18), max(95, avg_words + 40)
        else:
            low, high = max(35, int(avg_words * 0.8)), max(65, int(avg_words * 1.1))
    else:
        if is_content:
            low, high = max(45, avg_words), max(80, avg_words + 24)
        else:
            low, high = max(28, int(avg_words * 0.7)), max(50, int(avg_words * 0.95))

    if high <= low:
        high = low + 10
    return f"{low}-{high} words"


def slide_content_prompt(
    req: PresentationRequest,
    outline: SlideOutline,
    evidence_text: str,
    *,
    scenario_plan_text: str = "",
    min_bullets: int,
    max_bullets: int,
    target_total_seconds: int,
    voiceover_profile: str,
) -> str:
    """Build the strict JSON prompt used for per-slide content generation."""
    is_content = outline.type == "content"
    bullets_rule = f"{max(1, int(min_bullets))} to {max(1, int(max_bullets))}"
    voiceover_rule = _voiceover_rule(
        req=req,
        is_content=is_content,
        target_total_seconds=target_total_seconds,
        voiceover_profile=voiceover_profile,
    )
    scenario_block = ""
    if scenario_plan_text.strip():
        scenario_block = (
            "\nScenario plan to follow strictly:\n"
            f"{scenario_plan_text}\n"
        )
    style_block = _writing_style_block(os.environ.get("WRITING_STYLE", ""))
    return (
        "Generate the content for one presentation slide in English.\n"
        f"Topic: {req.topic}\n"
        f"Style: {req.style}\n"
        f"Slide number: {outline.slide_number}\n"
        f"Slide type: {outline.type}\n"
        f"Title draft: {outline.title}\n"
        f"Goal: {outline.goal}\n"
        f"{style_block}"
        "\nBook evidence excerpts:\n"
        f"{evidence_text}\n"
        f"{scenario_block}"
        "\nReturn JSON only with keys:\n"
        "- title\n"
        "- subtitle\n"
        f"- bullets ({bullets_rule})\n"
        "- voiceover\n"
        "- image_suggestion\n"
        "- layout_hint\n"
        "Rules:\n"
        "- Keep title very concise (prefer 3-7 words; hard limit 8 words) so it fits cleanly.\n"
        "- Add subtitle as one short contextual line (prefer 4-10 words).\n"
        "- Use facts grounded in the provided evidence.\n"
        "- Write from a teacher/professor perspective for undergraduate students.\n"
        "- Prioritize clarity over detail density; each bullet should express one teachable idea.\n"
        f"- voiceover should be {voiceover_rule}, clear spoken style.\n"
        "- Keep voiceover concise, natural, and easy to narrate in one breath per sentence.\n"
        "- Voiceover must explicitly reuse key terms from the bullets (do not paraphrase everything away).\n"
        "- Voiceover must include at least one concrete fact grounded in the Book evidence excerpts.\n"
        "- Voiceover must reuse at least two exact technical terms from the Book evidence excerpts.\n"
        "- For content slides, each bullet should carry a concrete technical fact, not generic wording.\n"
        "- For content slides, each bullet should be around 6-14 words.\n"
        "- Prefer fewer, stronger bullets instead of many short bullets.\n"
        "- image_suggestion must be an English diagram-oriented description (no filenames, no non-English text).\n"
        "- image_suggestion should mention what boxes/nodes/arrows should show for technical process slides.\n"
        "- layout_hint must be one of: title, text_only, text_left_image_right, section_divider, conclusion.\n"
        "- Keep bullets concise and concrete.\n"
    )
