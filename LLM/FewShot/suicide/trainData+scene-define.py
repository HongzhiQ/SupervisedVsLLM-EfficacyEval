# coding: utf-8
import openai
import pandas as pd
import time
import random

# your ChatGPT API key
openai.api_key = 'API-KEY'  # Please replace with your API key.

def get_classification(prompt, temperature=0.9):
    messages = [{"role": "user", "content": content} for content in prompt]
    response = openai.ChatCompletion.create(
        #Replace here with the GPT version you are using.
        model="gpt-4",
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message['content']

def get_examples(data, num_examples=12):
    categories = ["label"]
    examples = []

    for category in categories:
        positive_samples = data[data[category] == 1].sample(num_examples)
        negtive_samples = data[data[category] == 0].sample(num_examples)
        for _, row in positive_samples.iterrows():
            examples.append(f"{row['comment']},{row['label']}")
        for _, row in negtive_samples.iterrows():
            examples.append(f"{row['comment']},{row['label']}")

    return examples


data = pd.read_csv("../../../data/suicide/suicide_val_LLM.csv", encoding="utf-8", sep='\t')
train_data = pd.read_csv("../../../data/suicide/suicide_train_LLM.csv", encoding="utf-8", sep='\t')

BATCH_SIZE = 20
prompt_intro = ["下方内容分别显示了用户的ID和其在社交媒体上的发文内容。请你进行二分类任务，判断以下帖子是低自杀风险还是高自杀风险。请你用0、1标注高低自杀风险，0表示低自杀风险，1表示高自杀风险，请你用markdown表格的形式输出，输出结果的格式如下：表头依次为id、标签。注意：请只生成MarkDown表格，其他文字不要有。接下来我将给你几个段落以及其相对应的专家标注结果（注意：以下标签0代表低自杀风险，1代表高自杀风险），请你学习下面的例子。\n"]
prompt_act = ["\n请你根据学习到的专家标注例子，进行下述内容的判断:\n"]

#Replace here with the path to your result file.
with open('result.txt', 'a') as result_file:
    table_header = "| id | 标签 |"
    table_divider = "|----|------|"
    result_file.write(table_header + "\n")
    result_file.write(table_divider + "\n")

    for start_index in range(0, len(data), BATCH_SIZE):
        batch = data.iloc[start_index:start_index + BATCH_SIZE]
        prompts_for_batch = []

        for index, row in batch.iterrows():
            prompt = [f"id: {row['id']} {row['comment']}"]
            prompts_for_batch.extend(prompt)

        example_prompts = get_examples(train_data)
        complete_prompt = prompt_intro + example_prompts + prompt_act + prompts_for_batch
        print(complete_prompt)
        classification = get_classification(complete_prompt)

        results = classification.split("\n")[2:]
        print(results)
        for i, row in enumerate(batch.iterrows()):
            result_file.write(results[i] + "\n")

        time.sleep(30)

print("Finished!")
