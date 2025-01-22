from string import Template

MAX_TOKENS = 1024
GENERATION_TEMPERATURE = 1.0
GENERATION_SEED = 215

MULTI_CHOICE_COT_PROMPT = Template("""
Question: $question
$optionized_str

Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: 'Therefore, the final answer is: $$LETTER' (without quotes), where $$LETTER is one of the options. Think step by
step before answering.""")

OPEN_ENDED_COT_PROMPT = Template("""
Question: $question

Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: 'Therefore, the final answer is: 'Answer: $$ANSWER' (without quotes), where $$ANSWER is the final answer of the question. Think step by step
before answering.""")

MULTI_CHOICE_DO_PROMPT = Template("""
Question: $question
$optionized_str

Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices.
""")

OPEN_ENDED_DO_PROMPT = Template("""
Question: $question

Do not generate any intermediate reasoning process. Directly output the final short answer.
""")

COT_PROMPT = {
    "multiple-choice": MULTI_CHOICE_COT_PROMPT,
    "open-ended": OPEN_ENDED_COT_PROMPT
}

DO_PROMPT = {
    "multiple-choice": MULTI_CHOICE_DO_PROMPT,
    "open-ended": OPEN_ENDED_DO_PROMPT
}