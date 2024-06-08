import re 
import json 
prompt_template_ke_with_choices = """
Here is a math question: "{input}"
Correct answer is among: A: {A}, B: {B}, C: {C}, D: {D}.
Let's analyze the question from the following angles, print out each rationals in each step:
1. Read question and choices carefully.
2. According to math education syllabus, what category does the question belong to?
3. What domain specific problem solving skills and knowledge are commonly used to solve questions of the category?
4. Select the most suitable method to solve the question.
5. Solve the question step by step, pay attention to make use of information in both question and choices. 
6. print out final result, must in format "the answer is _final_result_" in the last line where _final_result_ is one of "(A)", "(B)", "(C)", "(D)" and "(None)", without any other text. 
"""


prompt_template_ke_with_choices = """
Here is a math question: "{input}"
Correct answer is among: A: {A}, B: {B}, C: {C}, D: {D}.
Let's analyze the question from the following angles, print out each rationals in each step:
1. Read question and choices carefully.
2. According to math education syllabus, what category does the question belong to?
3. What domain specific problem solving skills and knowledge are commonly used to solve questions of the category?
4. Select the most suitable method to solve the question.
5. Solve the question step by step, pay attention to make use of information in both question and choices. 
6. Compare answer against the choices choice_A: {A}, choice_B: {B}, choice_C: {C}, choice_D: {D}, and decide which choice is selected. If answer matches a choice, select the choice as final result; if answer doesn't match any choice, the answer is not correct, and final result is "None".
7. print out final result in format "Answer: the final result" in the last line, where the final result is one of "(A)", "(B)", "(C)", "(D)" and "(None)", without any other text. 
"""
prompt_template_ke_with_choices = """
Here is a math question: "{input}"
Correct answer is among: choice_A: {A}, choice_B: {B}, choice_C: {C}, choice_D: {D}.
Let's analyze the question from the following angles, print out each rationals in each step:
1. Read question and choices carefully.
2. According to math education syllabus, what category does the question belong to?
3. What domain specific problem solving skills and knowledge are commonly used to solve questions of the category?
4. Select the most suitable method to solve the question.
5. Solve the question step by step, pay attention to make use of information in both question and choices. 
6. Compare answer against the choices choice_A: {A}, choice_B: {B}, choice_C: {C}, choice_D: {D}, and decide which choice is selected. If answer matches a choice, select the choice as final result; if answer doesn't match any choice, the answer is not correct, and final result is "None".
7. print out final result in format "Answer: the final result" in the last line, without any other text. 
"""

prompt_template_ke_with_choices_20240406 = """
Here is a math question: "{input}"
Correct answer is among: A: {A}, B: {B}, C: {C}, D: {D}.
Let's analyze the question from the following angles, print out each rationals in each step:
1. According to math education syllabus, what category does the question belong to?
2. What domain specific knowledge and problem solving skills are suitable for solving this question?
3. Solve the question step by step, make sure every step is mathematically solid. 
4. Select final result from A: {A}, B: {B}, C: {C}, D: {D}. If answer doesn't match any choice, the final result is "None".
5. print out final result in format "Answer: the final result" in the last line, without any other text. 
"""





mmlu_prompt = json.load(open('MMLU/lib_prompt/mmlu-cot.json'))

def escape_curly_braces(s):
    # Escape opening curly braces
    s = s.replace("{", "{{")
    # Escape closing curly braces
    s = s.replace("}", "}}")
    return s

def prompt_template_few_shot_cot(task_name):

    return escape_curly_braces(mmlu_prompt[task_name]) + \
        """
        Q: {input}
        Answer Choices: (A) {A} (B) {B} (C) {C} (D) {D}
        A: Let's think step by step. 
        """
#
# def prompt_template_few_shot_cot_different_category(task_name):
#     return escape_curly_braces(mmlu_prompt['high_school_statistics']) + \
#         """
#         {input}
#         (A) {A} (B) {B} (C) {C} (D) {D}
#         A: Let's think step by step. 
#         """


def compose_few_shot_prompt(task_name, input, A, B, C, D):
    return prompt_template_few_shot_cot(task_name).format(input=input, A=A, B=B, C=C, D=D)

prompt_template_zero_shot_cot = """
Q: {input}
Answer Choices: (A) {A} (B) {B} (C) {C} (D) {D}
A: Let's think step by step. Print out each step, and output the selected choice from A, B, C and D in format "the answer is (selected choice)" without any other text.
"""


prompt_template_zero_shot_cot_forget_training_data = """
Q: {input}
Answer Choices: (A) {A} (B) {B} (C) {C} (D) {D}
A: Let's think step by step. If you saw the question in your training data, forget the training data and think on your own from scratch because the answer in your training data might be wrong.
Print out each step, output the selected choice from A, B, C and D in format "the answer is (selected choice)" without any other text.
"""


