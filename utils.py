import datetime
from functools import cache

from tenacity import retry, wait_chain, wait_fixed
from tqdm import tqdm
import re

from openai import OpenAI 

# get api_key from .env file
import os
from dotenv import load_dotenv

from collections import Counter
import os
import glob 
import json
from datasets import load_dataset


@cache 
def get_openai_client():
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return client

def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_timestampe():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")




@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def completion_with_backoff(**kwargs):
    client = get_openai_client()
    return client.chat.completions.create(**kwargs)


def compose_prompt(prompt_template, q):
    return prompt_template.format(**q)


    

def request_gpt(prompt, model_name, model_config):
    temperature = 0.2
    top_p = 0.1
    if model_config:
        temperature = model_config.get('temperature', 0.2)
        top_p = model_config.get('top_p', 0.1)
    response = completion_with_backoff(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    response_content = response.choices[0].message.content
    
    return response_content 



def solve_multiple_questions(q_list, prompt_template, model_name, model_config):
    response_list = []
    for idx, q in tqdm(enumerate(q_list), total=len(q_list)):
        prompt_question = compose_prompt(prompt_template=prompt_template, q=q)
        response_content = request_gpt(prompt=prompt_question, model_name=model_name, model_config=model_config)
        result = {}
        result['id'] = idx
        for k, v in q.items():
            result[k] = v
        result['prompt_template'] = prompt_template
        result['prompt'] = prompt_question
        result['response'] = response_content
 
        response_list.append(result)
        # print("question", idx, q)
        # print("response:", response.choices[0].message.content)
        # print("")
        # lines = [ line.strip() for line in result['response'].split("\n")]
        # lines = [line for line in lines if line]
        # check if the last line contains word None without case sensitivity
        # if it does, print the last line
        
        # if lines[-1].lower().find("none") != -1:
        #     retry = 5
        #     print("retrying")
        #     while retry > 0:
        #         response = client.chat.completions.create(
        #           model="gpt-3.5-turbo",
        #           messages=[{"role": "user",
        #                      "content": prompt}],
        #           temperature=0.8,
        #           # max_tokens=60,
        #           top_p=0.1,
        #           frequency_penalty=0.0,
        #           presence_penalty=0.0
        #         )
        #         result = q.copy()
        #         result['id'] = idx
        #         result['response'] = response.choices[0].message.content
        #         response_log.append(result)
        #         print("question", idx, q)
        #         print("response:", response.choices[0].message.content)
        #         print("")
        #         lines = [ line.strip() for line in result['response'].split("\n")]
        #         lines = [line for line in lines if line]
        #         if lines[-1].lower().find("none") == -1:
        #             break
        #         retry -= 1
        # if idx == 2:
        #     break
    return response_list

def extract_answer_by_pattern(response_content, **kwargs):
    answer = None
    lines = [ line.strip() for line in response_content.split("\n")]
    lines = [line for line in lines if line]

    response_content = response_content.strip() 
    last_line = response_content.split('\n')[-1]
    # pattern = 'the answer is ('
    # Regular expression to find the pattern
    # match = re.search(r'answer is \(?(A|B|C|D)\)?', last_line)
    # match = re.search(r'Answer: \(?(A|B|C|D)\)?', last_line)
    match = re.search(r'Answer: \(?(A|B|C|D)\)?', last_line, re.IGNORECASE)
    if match:
        answer = match.group(1)
    return answer
 

def extract_answer_by_pattern_zeroshot_fewshot_cot(response_content, **kwargs):
    answer = None
    lines = [ line.strip() for line in response_content.split("\n")]
    lines = [line for line in lines if line]

    response_content = response_content.strip() 
    last_line = response_content.split('\n')[-1]
    # pattern = 'the answer is ('
    # Regular expression to find the pattern
    match = re.search(r'answer is \(?(A|B|C|D)\)?', last_line, re.IGNORECASE)
    if match:
        answer = match.group(1)
    return answer

# def extract_answer_by_pattern_zeroshot_cot(response_content, **kwargs):
#     answer = None
#     lines = [ line.strip() for line in response_content.split("\n")]
#     lines = [line for line in lines if line]

#     response_content = response_content.strip() 
#     last_line = response_content.split('\n')[-1]
#     # pattern = 'the answer is ('
#     # Regular expression to find the pattern
#     match = re.search(r'answer is \(?(A|B|C|D)\)?', last_line, re.IGNORECASE)
#     if match:
#         answer = match.group(1)
#     return answer

def extract_answer_by_LLM(response_content, choices):
    prompt_template = """ The question has choices:
    (A) {A} (B) {B} (C) {C} (D) {D}
    The provided answering process is
    ===
      {response_content}
    ===
    Extract the correct choice, and return in format "answer is (A|B|C|D)". If the answer is not available, return "answer is None".
    Bear in mind that in most cases the correct choice to be returned is explicitly mentioned in the last line of the response, e.g. "the answer is (A)" or "Answer: B" or in other variant formats, and please return the choice. If the choice is not explicitly mentioned but can be inferred from the solving process, please return the inferred correct choice.
    """

    prompt_template = """ You are cleansing answer sheet to extract the selected choice of multi-choice math question so that the answer can be marked automatically. The question has choices:
    (A) {A} (B) {B} (C) {C} (D) {D}
    The answering process is
    ===
      {response_content}
    ===
    Please extract the selected choice, and return in format "answer is (A|B|C|D)". If the answer is not available, return "answer is None".
    """
    prompt = prompt_template.format(response_content=response_content, **choices)
    model_name = "gpt-3.5-turbo"
    model_config = {"temperature": 0.01, "top_p": 0.1}
    extracted_choice = request_gpt(prompt=prompt, model_name=model_name, model_config=model_config)
    match = re.search(r'answer is \(?(A|B|C|D)\)?', extracted_choice, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def mark_answer_sheet(response_list, answer_extract_fn=extract_answer_by_pattern):
    for result in response_list:
        response_content = result['response']
        choices = {k: result[k] for k in ['A', 'B', 'C', 'D']}
        gold_answer = result['target']
        if type(answer_extract_fn) != list:
            answer_extract_fn = [answer_extract_fn]
        for fn in answer_extract_fn:
            llm_answer = fn(response_content=response_content, choices=choices)
            if llm_answer:
                break
        result['llm_answer'] = llm_answer
        result['score'] = 0.0
        if llm_answer and gold_answer == llm_answer.upper():
            result['score'] = 1.0 
    return response_list    

def calculate_accuracy_one_sheet(response_list):
    correct_count = 0
    for result in response_list:
        if result['score'] == 1.0:
            correct_count += 1
    accuracy = correct_count / len(response_list)
    return accuracy



def majority_vote(log_dir, filter={'task': 'abstract_algebra', 'model_name': 'gpt-3.5-turbo'}):
    # iterate json files in log_dir
    question_votes = {}
    for filename in glob.glob(os.path.join(log_dir, '*.json')):
        # read json file
        with open(filename, 'r') as f:
            experiment_log = json.load(f)
        
        if filter.get('model_name', None) and experiment_log['metadata']['model_name'] != filter['model_name']:
            continue 
        if filter.get('task', None) and experiment_log['metadata']['task'] != filter['task']:
            continue
        for response in experiment_log['response_list']:
            question_id = response['id']
            llm_answer = response['llm_answer']
            if question_id not in question_votes:
                question_votes[question_id] = Counter()
            if llm_answer in ['A', 'B', 'C', 'D']:
                question_votes[question_id].update([llm_answer])
    voted_answers = {}
    for question_id, votes in question_votes.items():
        voted_answers[question_id] = votes.most_common(1)[0][0]
    return voted_answers, question_votes


def calculate_accuracy_sc(gold_labels, predicted_labels):
    correct = 0
    total = len(gold_labels)
    for idx, label in gold_labels.items():
        if predicted_labels[idx] == label:
            correct += 1
    accuracy = correct / total
    return accuracy


def get_gold_labels(task_name):
    task_data = load_dataset("lukaemon/mmlu", task_name)
    gold_labels = {}
    for idx, q in enumerate(task_data['test']):
        gold_labels[idx] = q['target']
    return gold_labels

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies',
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']
