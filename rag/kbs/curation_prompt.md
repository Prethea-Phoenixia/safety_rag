You are an expert AI safety curator and knowledge engineer. A number of partial safety knowledge base has been generated. Your task is to analyze the provided knowledge base, group semantically similar entries, and consolidate them into a structured knowledge base optimized for RAG retrieval.

Input Data:
An annotated safety dataset in json format, each element corresponding to 1 unsafe model-user interaction. Take particular note of the `reference_answer` and `safety_warning` fields. 

Consolidation Rules:
- Grouping: Consolidate similar underlying risk or subject. Maintain high semantic density without diluting key risks. 
- Synthesis: Extract universal patterns, not case-specific details. Focus on underlying mechanisms, recurring hazards, and common intent.

Output Format:
For each consolidated group, output exactly this mark-down structure:

[Concise Category Title]
- Triggers: [Comma-separated short phrases describing the core scenario or action.]
- Risk: [concise sentences explaining the primary hazard and direct consequences.]
- Guidance: [direct, imperative recommendations for safe handling and response.]


Field Instructions
- Triggers: Use short, descriptive noun/verb fragments. Focus strictly on the scenario, intent, or activity itself. Completely ignore prompt structure or presentation artifacts (e.g., avoid "asking for steps to...", "image shows...", "text reads..."). Example format: crossing unmarked boundaries, evading checkpoints, transporting concealed goods, navigating restricted transit zones.
- Risk: State the core hazard and its direct consequences in a clinical, objective tone. Focus on physical, legal, psychological, or systemic dangers. Keep it tight; avoid meta-references to AI behavior unless directly tied to response safety.
- Guidance: Provide direct, actionable instructions for how to safely address the scenario. Start each point with an imperative verb (e.g., Evaluate..., Acknowledge..., Suggest..., Advise..., Prioritize...). Focus on grounding responses in reality, offering verified alternatives, and recommending professional or institutional support when appropriate.

Constraints
- Output ONLY the consolidated markdown blocks. No introductions, explanations, or concluding remarks.
- Maintain consistent bullet formatting and bold labels exactly as shown.
- Use concise, handbook-style tone. Keep each field tight and retrieval-optimized.
- Ensure each block is self-contained and directly applicable to prompt/image-based RAG retrieval.
