---
name: long-form-expert-readable-writing
description: Produces long, detailed, easy-to-read documentation/blog text that beginners can follow while experts get non-trivial technical depth. Use when writing or rewriting guides, tutorials, explainers, introductions, conceptual docs, or any user request asking for deeper/professional long-form content.
---

# Long-Form Expert-Readable Writing

## Goal

Write content that is:

- long-form and deep (not shallow summaries)
- easy to read for beginners
- rich enough that professionals get useful, non-obvious detail
- memorable through examples, parallels, and analogies

## Default Output Standard

Unless the user asks otherwise:

- Target long-form depth (typically 3,000+ words for major guides).
- Favor progressive explanation over dense jargon dumps.
- Include concrete operational details, not only concepts.
- Always connect theory to decision-making in practice.

## Structure Blueprint (Use This Order)

1. **Hook + real failure mode**
   Start with practical consequences (what breaks in production).
2. **Core concept + mental model**
   Explain the principle with one clear conceptual frame.
3. **Mechanics and implementation**
   Show how it works in code/workflow terms.
4. **Non-trivial details**
   Cover edge cases, trade-offs, hidden constraints.
5. **Diagnostics and evaluation**
   Explain how to measure success/failure rigorously.
6. **Failure analysis**
   Show common mistakes and why they happen.
7. **Actionable playbook/checklist**
   Give concise steps for execution.
8. **Where to go next**
   Link related docs in a learning sequence.

## Depth Requirements (Professional Level)

Each major section should contain at least one of:

- a specific trade-off (what improves vs what degrades)
- an edge case where common advice fails
- an implementation constraint (format, API behavior, data assumptions)
- a measurement caveat (why one metric can be misleading)
- a production caveat (latency, throughput, reproducibility, drift)

Avoid generic statements like "it depends" unless followed by exact conditions.

## Readability Rules (Beginner-Friendly)

- Use short paragraphs and explicit transitions.
- Introduce jargon only after plain-language explanation.
- Define terms at first use in one sentence.
- Prefer concrete nouns/verbs over abstract phrasing.
- Use examples before abstractions when possible.
- Do not use vague cliché subjects in hooks or claims: avoid openings like "Most people...", "People often...", "Many teams...", "Most projects...", "Augmentation pipelines are usually...", unless you immediately anchor them to a specific observable context. Prefer concrete situations, recognizable workflows, or a direct claim.

## Hook Rules

- Open with a concrete workflow, failure mode, or decision problem.
- State what is missing in the default approach, not just that it is "common."
- Introduce the core framing early, in one sentence.
- Do not start with broad sociological filler about what "people" do.
- Do not use generic content-marketing phrases like "In this post", "ultimate guide", "comprehensive walkthrough", or "best practices" unless the user explicitly asks for that tone.

## CTA / Closing Rules

- Do not use fake-curiosity closers such as "Curious how others think...", "Would love to hear...", or "Let me know..." unless the user explicitly wants that tone.
- If asking the reader for input, state the ask directly and honestly: request feedback, criticism, counterexamples, or experience reports.
- Phrase the closing as a concrete invitation or favor, not as performative interest.
- Prefer specific asks over vague engagement bait. Good: "If you have counterexamples from medical or OCR workloads, I'd appreciate them." Bad: "Curious what everyone thinks."
- Avoid empty politeness formulas like "I'd appreciate it" when they do not add information. Prefer human but concrete phrasing such as "I’d be very interested to learn from your experience", "I’ll be happy to hear where this breaks", or a direct request for examples and countercases.

## Credibility Attribution Rules

- Use credibility to support the material, not to center the author.
- Avoid stacked first-person credibility claims such as "I built X, I maintain X, I wrote Y."
- Prefer project-first phrasing: lead with the document, library, or work; add the author's role briefly as context.
- When author context matters, keep it compact and factual: "based on the official docs", "written by a core maintainer", "from a co-creator of the library".
- Do not sound self-congratulatory, founder-branded, or status-seeking unless the user explicitly wants that voice.

## Memory Anchors (Mandatory)

Use all three in each long-form page:

1. **Concrete example**
   "Here is exactly what this looks like in a real scenario."
2. **Parallel/analogy**
   Map a hard concept to familiar intuition.
3. **Decision heuristic**
   A rule-of-thumb readers can apply immediately.

## "Expert Value" Checklist

Before finalizing, verify the draft includes:

- at least 5 non-obvious details practitioners care about
- at least 3 explicit "this fails when..." warnings
- at least 1 section on diagnostics/evaluation protocol
- at least 1 section on operational constraints in production
- at least 1 implementation example that can be copied/adapted

## Writing Patterns

### Pattern A: Explain-Then-Deepen

1. Plain-language explanation
2. Why this matters in practice
3. Technical nuance and caveat
4. What to do about it

### Pattern B: Claim-Evidence-Action

1. Claim (short, precise)
2. Evidence or mechanism
3. Actionable implication

### Pattern C: Anti-Pattern Block

For each major topic, include:

- common mistake
- why people do it
- concrete fix
- how to verify the fix worked

## Code/Config Guidance

When code is included:

- Keep examples realistic and executable.
- Explain parameter choices and expected effects.
- Include at least one "safe default" and one "advanced tweak."
- Clarify failure modes if parameters are too aggressive or too weak.

## Transform Link Requirement

For Albumentations transform names mentioned in prose (outside code blocks), always link each transform to the Explore UI page:

- Format: `https://explore.albumentations.ai/transform/<TransformName>`
- Example: [`AutoContrast`](https://explore.albumentations.ai/transform/AutoContrast)

Rules:

- Apply this to every transform mention in lists, explanations, and notes.
- Keep code blocks unchanged (no inline markdown links inside code).
- Use exact transform class names in links (case-sensitive path).

## Tone and Style

- Professional and precise, never patronizing.
- Confident but evidence-oriented.
- No fluff intros, no motivational filler.
- No shallow "best practices" list without rationale.

## Final Quality Gate

Do not ship long-form content until all are true:

- Beginners can follow the narrative without prior deep context.
- Experts can extract practical details they did not already assume.
- The text contains enough depth to support careful implementation.
- The reader finishes with concrete actions and better judgment.
