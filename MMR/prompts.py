# -*- coding: utf-8 -*-
"""Prompt templates used across the MemLoc pipeline.

All templates below are transcribed verbatim from Appendix J
("Prompt Templates") of the camera-ready paper:

    "Where to Look and What to Use: Retrieve-Localize-Generate for
     Long-Term Conversational Memory Question Answering" (EMNLP 2026)

In the paper, blue text denotes sample-specific inputs and black text the
fixed instructions.  Here, sample-specific inputs are written as
``{Placeholder}`` tokens that are substituted by :func:`render`.

Templates
---------
1. ``SUMMARIZE_AND_KEYWORDS``      -> Memory_Construct.py   (multi-granularity memory construction)
2. ``EVENT_EXTRACTION``            -> event_summary.py      (event-level memory construction)
3. ``EVIDENCE_FILTERING``          -> locator_inference.py (inner-memory extraction / cross-memory reranking)
4. ``MULTI_GRANULAR_GENERATION``   -> generation.py         (cue-guidance answer generation)
5. ``ANSWER_VERIFICATION``         -> generation_judge.py   (LLM-as-a-judge evaluation)
6. ``SFT_RL_TRAINING``             -> REL training data / EasyR1 format prompt
7. ``HINT_EXTRACTION``             -> REL/hint_shpo.py     (SHPO self-reflective hint extraction)
"""

import re

# =====================================================================
# 1. Prompt Template for Summarization and Keyword Extraction
#    (used in Memory_Construct.py to build summary / keyword memories)
# =====================================================================
SUMMARIZE_AND_KEYWORDS = """Instruction: You are an intelligent and insightful individual. Your task is to analyze the conversation between a user and an AI assistant and extract two key elements: a concise summary of the conversation and the most relevant keywords.

Input: {Conversation}

Tasks:
1. Summary: Provide a concise paragraph that summarizes the main topics and key information of the conversation.
2. Keywords: Extract the most relevant keywords from the conversation content.

Output Format:
Return a JSON object that must strictly contain the following structure:
{
    "memory":
    {
        "summary": "<A concise summary of the conversation>",
        "keywords": "<Keyword 1>; <Keyword 2>; <Keyword 3>; ..."
    }
}

Only provide the JSON object without any additional text.

Answer:"""

# =====================================================================
# 2. Prompt Template for Event Extraction and Timeline Construction
#    (used in event_summary.py to build event / temporal memories)
# =====================================================================
EVENT_EXTRACTION = """Instruction: You are an intelligent system designed to extract structured temporal information from conversations. Your task is to identify key events and construct a standardized timeline based on the given dialogue.

Input:
Conversation Date: {ConversationDate}
Conversation Content: {Conversation}

Tasks:
1. Key Event Extraction: Identify the key events mentioned in the conversation.
2. Timeline Construction: Organize the extracted events in chronological order.
3. Role Awareness: Summarize the information provided by the assistant and the user separately.

Requirements:
- Output only concise event summaries.
- Do NOT quote or restate the original dialogue.
- Ensure the results are clear, structured, and information-dense.

Output Format:

| Date | Key Event | Description |
|------|-----------|-------------|
| YYYY/MM/DD | xxx | xxx |
| YYYY/MM/DD | xxx | xxx |

Answer:"""

# =====================================================================
# 3. Prompt Template for Evidence Filtering and Sentence Selection
#    (used in locator_inference.py for inner-memory extraction and
#     cross-memory reranking)
# =====================================================================
EVIDENCE_FILTERING = """Instruction: You are an expert in information retrieval and evidence filtering. Your task is to identify the sentences that are most relevant to answering a given question based on retrieved documents.

Input:
Question: {Question}
Retrieved Documents: {Documents}

Tasks:
1. Analyze the relevance of each sentence to the question.
2. Select the most relevant sentences.
3. Provide a final answer based on the selected evidence.

Requirements:
- Perform step-by-step reasoning.
- Only select the most relevant sentences.
- Ensure the final answer is accurate and grounded in the selected evidence.

Output Format:

<reason>...step-by-step reasoning...</reason>
<id>Most relevant sentence IDs</id>
<answer>answer</answer>

Answer:"""

# =====================================================================
# 4. Prompt Template for Multi-Granular Reasoning and Answer Generation
#    (used in generation.py for cue-guidance generation)
# =====================================================================
MULTI_GRANULAR_GENERATION = """Instruction: You are an intelligent conversational assistant. Your task is to carefully analyze the provided conversation history and multi-granular contextual information, and generate a concise, accurate, coherent, and helpful answer to the given question.

Input:
Conversation History and Context: {Context}
Reference Information (optional): {Reference}
Question: {Question}

Tasks:
1. Understand the conversation history and contextual signals.
2. Integrate multi-granular information (e.g., session, summary, events, time).
3. Generate a coherent and accurate answer.

Requirements:
- The answer must be grounded primarily in the provided conversation history.
- Do NOT introduce unsupported or external information.
- Keep the answer concise and focused.
- Ensure logical coherence and readability.
- If the question involves time, provide explicit timestamps when possible.
- Prefer reusing original wording from the conversation when appropriate.
- Use reference information only as supporting clues, not as the sole basis.
- Answer must be in English.

Output Format:
Provide a direct and concise answer to the question.

Answer:"""

# =====================================================================
# 5. Prompt Template for Answer Verification
#    (used in generation_judge.py, LLM-as-a-judge evaluation)
# =====================================================================
ANSWER_VERIFICATION = """Instruction: Determine whether the model response correctly answers the question based on the reference answer.

Input:
Question: {Question}
Reference Answer: {Answer}
Model Response: {Response}

Requirement:
Output [[yes]] if the response is correct or equivalent to the reference answer; otherwise output [[no]].

Output Format:

[[yes]] or [[no]]

Answer:"""

# =====================================================================
# 6. Prompt Template for SFT & RL Training
#    (used to build the Locator training data; the same structured output
#     is enforced by REL/EasyR1-main/examples/format_prompt/answer.jinja)
# =====================================================================
SFT_RL_TRAINING = """Instruction: You are an expert in information retrieval and evidence filtering. Given a question and a list of retrieved documents, select the documents that are most relevant to answering the question. Output the corresponding document IDs, answer and explain your reasoning.

Input:
Question: {Question}
Retrieved Documents: {Documents}

Output Format:

<reason>...step-by-step reasoning...</reason>
<id>Most relevant document IDs</id>
<answer>answer</answer>"""

# =====================================================================
# 7. Prompt Template for Hint Extraction
#    (used in REL/hint_shpo.py, Self-reflective Hint Policy Optimization)
# =====================================================================
HINT_EXTRACTION = """Instruction: You are an intelligent and analytical teacher model. Your task is to analyze the divergence between correct reasoning trajectories and wrong reasoning trajectories, then extract a concise and domain-general corrective hint without leaking the ground-truth answer.

Input:
Question: {Question}
Correct Reasoning Trace: {CorrectTrace}
Wrong Reasoning Trace: {WrongTrace}
Answer: {Answer}

Tasks:
1. Identify Divergence: Determine which reasoning step caused the student model to deviate from the correct reasoning path.
2. Generate Hint: Produce a concise, domain-general strategy that can guide the student model back to the correct reasoning direction. The hint should focus on reasoning principles rather than specific answers.
3. Leakage Verification: Ensure that the generated hint does not reveal or directly imply the ground-truth answer.

Example Hints:

- When multiple timestamps conflict, prioritize the most recent entry as the user's current state.
- Verify whether the event occurred or was merely planned. Distinguish intent from completion.
- Disregard assistant-generated summaries when they contradict explicit user statements.
- Cross-check person attributes (age, location) across sessions; use the latest update.

Output Format:
Return a JSON object that must strictly contain the following structure:
{
    "hint":
    {
        "divergence_step": "<Reasoning step causing divergence>",
        "strategy_hint": "<Domain-general corrective hint>"
    }
}

Only provide the JSON object without any additional text.

Answer:"""


# =====================================================================
# Rendering helpers
# =====================================================================
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render(template: str, **kwargs) -> str:
    """Substitute ``{Placeholder}`` tokens in a template.

    Only tokens whose name is passed as a keyword argument are replaced,
    so literal braces inside the templates (e.g. the JSON examples) are
    never touched.
    """
    mapping = {key: "" if value is None else str(value) for key, value in kwargs.items()}

    def _sub(match: "re.Match") -> str:
        return mapping.get(match.group(1), match.group(0))

    return _PLACEHOLDER_RE.sub(_sub, template)


def build_evidence_filtering_prompt(question: str, documents: str) -> str:
    """Appendix J: Prompt Template for Evidence Filtering and Sentence Selection."""
    return render(
        EVIDENCE_FILTERING,
        Question=question,
        Documents=documents,
    )


def build_generation_prompt(
    context: str,
    reference: str,
    question: str,
    question_date: str = "",
) -> str:
    """Appendix J: Prompt Template for Multi-Granular Reasoning and Answer Generation.

    Optional ``question_date`` is folded into the ``{Question}`` slot so
    temporal datasets (LongMemEval / Long-MT-Bench+) keep their date cue
    without diverging from the paper template.
    """
    question_block = question
    if question_date:
        question_block = f"Question Date: {question_date}\nQuestion: {question}"
    return render(
        MULTI_GRANULAR_GENERATION,
        Context=context,
        Reference=reference or "None",
        Question=question_block,
    )
