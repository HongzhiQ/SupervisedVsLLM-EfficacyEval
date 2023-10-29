# coding: utf-8
import openai
import pandas as pd
import time

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

data = pd.read_csv("../../../data/suicide/suicide_val_LLM.csv", encoding="utf-8", sep='\t')


prompt_intro = ["高自杀风险的判断标准为：（1）自杀可能正在进行中。（2）自杀方式已确定，近日内可能进行。（3）自杀已经计划中，自杀日期大体确定。（4）自杀方式已确定，自杀日期未明。（5）自杀已经计划中，自杀日期未明。\n低自杀风险的判断标准为：（1）强烈的自杀愿望，自杀方式未明。（2）自杀愿望已表达，具体方式和计划未明。（3）强烈的生存痛苦，未见自杀愿望表达。（4）生存痛苦已明确表达，未见自杀愿望表达。（5）生存病苦有所表达，未见自杀愿望表达。（6）未见任何生存痛苦表达。\n下方内容分别显示了用户的ID和其在社交媒体上的发文内容。你是心理学专家，请你考虑下述帖子中表达的心理健康状况，并结合上述高自杀风险和低自杀风险的判断标准，进行二分类任务，判断以下段落是低自杀风险还是高自杀风险。请你用0、1标注高低自杀风险，0表示低自杀风险，1表示高自杀风险，请你用markdown表格的形式输出，输出结果的格式如下：表头依次为id、标签。注意：请只生成MarkDown表格，其他文字不要有。\n"]

BATCH_SIZE = 40

#Replace here with the path to your result file.
with open('result.txt', 'a') as result_file:
    table_header = "| id | 标签 |"
    table_divider = "|----|------|"
    result_file.write(table_header)
    result_file.write("\n")
    result_file.write(table_divider)
    result_file.write("\n")

    for start_index in range(0, len(data), BATCH_SIZE):
        batch = data.iloc[start_index:start_index + BATCH_SIZE]
        prompts_for_batch = []

        for index, row in batch.iterrows():
            prompt = [f"id: {row['id']} {row['comment']}"]
            prompts_for_batch.extend(prompt)

        complete_prompt = prompt_intro + prompts_for_batch
        print(complete_prompt)
        classification = get_classification(complete_prompt)

        results = classification.split("\n")[2:]  # Exclude header and separator rows.
        print(results)
        for i, row in enumerate(batch.iterrows()):
            result_file.write(results[i] + "\n")

        time.sleep(30)

print("Finished!")
