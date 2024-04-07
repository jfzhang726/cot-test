import datetime
from functools import cache

from tenacity import retry, wait_chain, wait_fixed
from tqdm import tqdm
import re

from openai import OpenAI 

# get api_key from .env file
import os
from dotenv import load_dotenv

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

def extract_answer_by_pattern(response_content):
    answer = None
    lines = [ line.strip() for line in response_content.split("\n")]
    lines = [line for line in lines if line]

    response_content = response_content.strip() 
    last_line = response_content.split('\n')[-1]
    # pattern = 'the answer is ('
    # Regular expression to find the pattern
    # match = re.search(r'answer is \(?(A|B|C|D)\)?', last_line)
    # match = re.search(r'Answer: \(?(A|B|C|D)\)?', last_line)
    match = re.search(r'Answer: \(?(A|B|C|D)\)?', last_line)
    if match:
        answer = match.group(1)
    return answer
 

def extract_answer_by_pattern_fewshot_cot(response_content):
    answer = None
    lines = [ line.strip() for line in response_content.split("\n")]
    lines = [line for line in lines if line]

    response_content = response_content.strip() 
    last_line = response_content.split('\n')[-1]
    # pattern = 'the answer is ('
    # Regular expression to find the pattern
    match = re.search(r'answer is \(?(A|B|C|D)\)?', last_line)
    # match = re.search(r'Answer: \(?(A|B|C|D)\)?', last_line)
    # match = re.search(r'Answer: \(?(A|B|C|D)\)?', last_line)
    if match:
        answer = match.group(1)
    return answer

def extract_answer_by_pattern_zeroshot_cot(response_content):
    answer = None
    lines = [ line.strip() for line in response_content.split("\n")]
    lines = [line for line in lines if line]

    response_content = response_content.strip() 
    last_line = response_content.split('\n')[-1]
    # pattern = 'the answer is ('
    # Regular expression to find the pattern
    match = re.search(r'answer is \(?(A|B|C|D)\)?', last_line)
    if match:
        answer = match.group(1)
    return answer

def extract_answer_by_LLM(response_content):
    # to be implemented
    return None

def mark_answer_sheet(response_list, answer_extract_fn=extract_answer_by_pattern):
    for result in response_list:
        response_content = result['response']
        gold_answer = result['target']
        llm_answer = answer_extract_fn(response_content)
        result['llm_answer'] = llm_answer
        result['score'] = 0.0
        if llm_answer and gold_answer == llm_answer:
            result['score'] = 1.0 
    return response_list    

def calculate_accuracy(response_list):
    correct_count = 0
    for result in response_list:
        if result['score'] == 1.0:
            correct_count += 1
    accuracy = correct_count / len(response_list)
    return accuracy


