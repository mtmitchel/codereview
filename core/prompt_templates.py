import os

# Constants
PROMPT_FILE_PATH = os.path.join(os.path.expanduser("~"), ".code_reviewer_prompts.json")
DEFAULT_PROMPT_NAME = "Default"

# --- Structured Output Schema --- (Informational, not part of the prompt itself)
# { "issues": [ { "line_start": <int>, "line_end": <int>, 
#                  "severity": "Critical | High | Medium | Low | Info", 
#                  "description": "<string>", "recommendation": "<string>" } ] }

# Enhanced Prompt Templates with Context-Enriched and Chain-of-Thought Prompting
DEFAULT_PROMPT_TEMPLATE = (
    "Analyze the following code snippet from the file '{file_path}' for potential issues, including but not limited to: \n"
    "- Bugs and logical errors\n"
    "- Security vulnerabilities\n"
    "- Performance bottlenecks\n"
    "- Adherence to best practices and coding standards\n"
    "- Code clarity, readability, and maintainability\n"
    "- Potential improvements and refactoring opportunities\n\n"
    "**File Context:**\n"
    "- File Path: {file_path}\n"
    "- Dependencies (if available): {dependencies}\n"
    "- Static Analysis (e.g., Flake8): {flake8_output}\n"
    "- Plugin Analysis (if available):\n"
    "  - Security Issues: {security_issues}\n"
    "  - Semantic Issues: {semantic_issues}\n"
    "  - Other Knowledge: {knowledge}\n\n"
    "**Code Snippet:**\n```{code_chunk}```\n\n"
    "**Instructions:**\n"
    "1. Review the **Code Snippet** considering the **File Context**. Identify all potential issues.\n"
    "2. For EACH issue found, provide the following details:\n"
    "   - `line_start`: The starting line number of the code related to the issue.\n"
    "   - `line_end`: The ending line number of the code related to the issue.\n"
    "   - `severity`: Classify the severity as one of: **Critical, High, Medium, Low, Info**.\n"
    "   - `description`: A concise explanation of the issue.\n"
    "   - `recommendation`: A specific suggestion on how to fix or improve the code.\n"
    "3. Format your entire response as a single JSON object containing a single key `\"issues\"` which holds a list of the issue objects described above.\n"
    "4. **IMPORTANT:** Output ONLY the raw JSON object. Do not include any introductory text, explanations, apologies, markdown formatting, or anything else before or after the JSON structure.\n"
    "5. If NO issues are found, return a JSON object with an empty list: `{{\"issues\": []}}`\n"
    "**JSON Output:**\n"
)

SECURITY_ANALYSIS_PROMPT_TEMPLATE = (
    "You are a security expert focused on code vulnerabilities. Analyze the following code for security issues "
    "including injection vulnerabilities, authentication problems, data exposure risks, and OWASP Top 10 concerns.\n\n"
    "File Path: {file_path}{dependencies}\n\n"
    "Code Chunk:\n```{code_chunk}```\n{flake8_output}\n\n"
    "**Instructions:**\n"
    "1. Review the code for security vulnerabilities.\n"
    "2. For EACH issue found, provide the following details:\n"
    "   - `line_start`: The starting line number of the code related to the issue.\n"
    "   - `line_end`: The ending line number of the code related to the issue.\n"
    "   - `severity`: Classify the severity as one of: **Critical, High, Medium, Low, Info**.\n"
    "      - Critical: Immediate exploitation risk with severe impact\n"
    "      - High: Serious security flaw that should be fixed soon\n"
    "      - Medium: Security issue that should be addressed in regular development\n"
    "      - Low: Minor security concern or best practice violation\n"
    "      - Info: Security observation or recommendation\n"
    "   - `description`: A concise explanation of the security vulnerability.\n"
    "   - `recommendation`: A specific suggestion on how to fix the security issue.\n"
    "3. Format your entire response as a single JSON object containing a single key `\"issues\"` which holds a list of the issue objects described above.\n"
    "4. **IMPORTANT:** Output ONLY the raw JSON object. Do not include any introductory text, explanations, or anything else.\n"
    "5. If NO security issues are found, return a JSON object with an empty list: `{\"issues\": []}`\n"
    "**JSON Output:**\n"
)

PERFORMANCE_ANALYSIS_PROMPT_TEMPLATE = (
    "You are a performance optimization expert. Analyze the following code for performance issues "
    "including inefficient algorithms, resource leaks, unnecessary operations, and scaling concerns.\n\n"
    "File Path: {file_path}{dependencies}\n\n"
    "Code Chunk:\n```{code_chunk}```\n{flake8_output}\n\n"
    "**Instructions:**\n"
    "1. Review the code for performance issues and inefficiencies.\n"
    "2. For EACH issue found, provide the following details:\n"
    "   - `line_start`: The starting line number of the code related to the issue.\n"
    "   - `line_end`: The ending line number of the code related to the issue.\n"
    "   - `severity`: Classify the severity as one of: **Critical, High, Medium, Low, Info**.\n"
    "      - Critical: Severe performance bottleneck with significant impact\n"
    "      - High: Significant performance issue that should be addressed soon\n"
    "      - Medium: Performance improvement opportunity for regular development\n"
    "      - Low: Minor performance enhancement possibility\n"
    "      - Info: Performance observation or suggestion\n"
    "   - `description`: A concise explanation of the performance issue.\n"
    "   - `recommendation`: A specific suggestion on how to improve performance.\n"
    "3. Format your entire response as a single JSON object containing a single key `\"issues\"` which holds a list of the issue objects described above.\n"
    "4. **IMPORTANT:** Output ONLY the raw JSON object. Do not include any introductory text, explanations, or anything else.\n"
    "5. If NO performance issues are found, return a JSON object with an empty list: `{\"issues\": []}`\n"
    "**JSON Output:**\n"
)

CROSS_FILE_PROMPT_TEMPLATE = (
    "You are an expert code reviewer focusing on system architecture and integration. Analyze the relationship between these two files.\n\n"
    "File 1: {file_path_a}\nSummary 1:\n```\n{summary_a}\n```\n"
    "File 2: {file_path_b}\nSummary 2:\n```\n{summary_b}\n```\n\n"
    "Focus your analysis on these specific areas, using clear headers:\n\n"
    "**1. Dependency Relationship:**\n"
    "- Does one file import/use the other? How and why?\n"
    "- Is there data/control flow between them? In what direction?\n\n"
    "**2. Interface Compatibility:**\n"
    "- Are there function signatures, APIs, or data structures shared between files?\n"
    "- Are there potential mismatches in how they interact?\n\n"
    "**3. Architectural Concerns:**\n"
    "- Are there circular dependencies or unnecessarily tight coupling?\n"
    "- Do these files follow consistent architectural principles?\n\n"
    "**4. Improvement Opportunities:**\n"
    "- Are there specific refactoring suggestions to improve their relationship?\n"
    "- Could the interaction between these files be made clearer or more robust?\n\n"
    "Be extremely concise. Prefer bullet points over paragraphs. If any section isn't relevant, write 'None identified' rather than speculating.\n\n"
    "If you find an issue, clearly state WHY it's a problem and provide a specific recommendation.\n"
)

SUMMARIZATION_PROMPT_TEMPLATE = (
    "You are an expert code analyst. Analyze the following code file and provide a concise summary with structured headers. "
    "Follow this specific format:\n\n"
    "1. **Primary Purpose:** Identify the main goal/responsibility of the file in 1-2 sentences.\n"
    "2. **Key Components:** Bullet list the most important functions, classes, or components with one-line descriptions.\n"
    "3. **External Dependencies:** List the main libraries, frameworks, or modules this file relies on.\n"
    "4. **Design Patterns/Approaches:** Note any architecture patterns, coding paradigms, or design approaches used.\n\n"
    "File Path: {file_path}\nDependencies: {dependencies}\n\n"
    "Code:\n```\n{code_content}\n```\n{flake8_output}\n\n"
    "Provide a structured analysis using the headers above. For each section, be extremely concise and avoid unnecessary words. "
    "The entire summary should be readable in under 30 seconds.\n\n"
    "Finally, conclude with a **Concise Summary (3-4 sentences)** that captures the file's essence.\n\n"
    "BE VERY BRIEF. Focus on what's truly important, not exhaustive coverage.\n"
)

DESIGN_PATTERN_ANALYSIS_PROMPT_TEMPLATE = (
    "You are an expert in software design patterns. Analyze the following code for design pattern usage, "
    "anti-patterns, and architectural concerns.\n\n"
    "File Path: {file_path}{dependencies}\n\n"
    "Code Chunk:\n```{code_chunk}```\n{flake8_output}\n\n"
    "**Instructions:**\n"
    "1. Review the code for design patterns, anti-patterns, and architectural concerns.\n"
    "2. For EACH issue or observation found, provide the following details:\n"
    "   - `line_start`: The starting line number of the code related to the issue/observation.\n"
    "   - `line_end`: The ending line number of the code related to the issue/observation.\n"
    "   - `severity`: Classify the severity as one of: **Critical, High, Medium, Low, Info**.\n"
    "      - Critical: Severe architectural or design flaw\n"
    "      - High: Significant design issue that impacts maintainability\n"
    "      - Medium: Design improvement opportunity\n"
    "      - Low: Minor design enhancement suggestion\n"
    "      - Info: Design pattern observation or suggestion\n"
    "   - `description`: A concise explanation of the design issue or pattern observation.\n"
    "   - `recommendation`: A specific suggestion for improvement if applicable.\n"
    "3. Format your entire response as a single JSON object containing a single key `\"issues\"` which holds a list of the issue objects described above.\n"
    "4. **IMPORTANT:** Output ONLY the raw JSON object. Do not include any introductory text, explanations, or anything else.\n"
    "5. If NO design issues are found, return a JSON object with an empty list: `{\"issues\": []}`\n"
    "**JSON Output:**\n"
)

TECHNICAL_DEBT_ANALYSIS_PROMPT_TEMPLATE = (
    "You are a technical debt expert. Analyze the following code for technical debt indicators, "
    "including code smells, maintainability issues, and opportunities for refactoring.\n\n"
    "File Path: {file_path}{dependencies}\n\n"
    "Code Chunk:\n```{code_chunk}```\n{flake8_output}\n\n"
    "**Instructions:**\n"
    "1. Review the code for technical debt indicators like code smells, duplication, complexity, etc.\n"
    "2. For EACH issue found, provide the following details:\n"
    "   - `line_start`: The starting line number of the code related to the issue.\n"
    "   - `line_end`: The ending line number of the code related to the issue.\n"
    "   - `severity`: Classify the severity as one of: **Critical, High, Medium, Low, Info**.\n"
    "      - Critical: Severe technical debt that blocks further development\n"
    "      - High: Significant debt that should be addressed soon\n"
    "      - Medium: Technical debt that should be addressed in regular development\n"
    "      - Low: Minor technical debt concern\n"
    "      - Info: Technical observation or refactoring opportunity\n"
    "   - `description`: A concise explanation of the technical debt issue.\n"
    "   - `recommendation`: A specific suggestion on how to refactor or improve the code.\n"
    "3. Format your entire response as a single JSON object containing a single key `\"issues\"` which holds a list of the issue objects described above.\n"
    "4. **IMPORTANT:** Output ONLY the raw JSON object. Do not include any introductory text, explanations, or anything else.\n"
    "5. If NO technical debt issues are found, return a JSON object with an empty list: `{\"issues\": []}`\n"
    "**JSON Output:**\n"
)

# Simpler template for when the entire file is processed as one chunk
SINGLE_CHUNK_PROMPT_TEMPLATE = (
    "Review the following code from the file '{file_path}'.\n"
    "Focus on identifying potential issues like bugs, security vulnerabilities, performance concerns, and areas for improvement in clarity or maintainability.\n\n"
    "**Code:**\n```{code_chunk}```\n\n"
    "**Instructions:**\n"
    "1. Review the **Code**. Identify all potential issues.\n"
    "2. For EACH issue found, provide the following details:\n"
    "   - `line_start`: The starting line number of the code related to the issue.\n"
    "   - `line_end`: The ending line number of the code related to the issue.\n"
    "   - `severity`: Classify the severity as one of: **Critical, High, Medium, Low, Info**.\n"
    "   - `description`: A concise explanation of the issue.\n"
    "   - `recommendation`: A specific suggestion on how to fix or improve the code.\n"
    "3. Format your entire response as a single JSON object containing a single key `\"issues\"` which holds a list of the issue objects described above.\n"
    "4. **IMPORTANT:** Output ONLY the raw JSON object. Do not include any introductory text, explanations, apologies, markdown formatting, or anything else before or after the JSON structure.\n"
    "5. If NO issues are found, return a JSON object with an empty list: `{{\"issues\": []}}`\n"
    "**JSON Output:**\n"
)
