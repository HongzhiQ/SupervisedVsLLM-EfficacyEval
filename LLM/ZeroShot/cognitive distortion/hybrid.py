# coding: utf-8
import openai
import pandas as pd
import time

# your ChatGPT API key
openai.api_key = 'API-KEY'  # Please replace with your API key.

def get_classification(prompt, temperature=0.1):
    messages = [{"role": "user", "content": content} for content in prompt]
    response = openai.ChatCompletion.create(
        #Replace here with the GPT version you are using.
        model="gpt-4",
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message['content']

data = pd.read_csv("../../data/cognitive_distortion_val_LLM.csv", encoding="gbk")

# hybrid prompting
prompt_intro = ["下方内容分别显示了用户的ID和其在社交媒体上的发文内容。你是一名心理学专家，请你考虑下述帖子中表达的心理健康状况，并进行多分类任务，判断以下帖子是否包含如下12种（非此即彼、以偏概全、心理过滤、否定正面思考、读心术、先知错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他）认知歪曲特征，请你用0和1标注认知歪曲特征，0表示不含有该认知歪曲特征，1表示含有该认知歪曲特征。请使用MarkDown表格形式输出分类结果。输出结果的格式如下：表头依次为id、非此即彼、以偏概全、心理过滤、否定正面思考、读心术、先知错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他。注意：请只生成MarkDown表格，其他文字不要有。"]

BATCH_SIZE = 20

#Replace here with the path to your result file.
with open('result.txt', 'a') as result_file:
    table_header = "| id | 非此即彼 | 以偏概全 | 心理过滤 | 否定正思考 | 读心术 | 先知错误 | 放大 | 情绪化推理 | 应该式 | 乱贴标签 | 罪责归己 | 罪责归他 |"
    table_divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    result_file.write(table_header)
    result_file.write("\n")
    result_file.write(table_divider)
    result_file.write("\n")

    for start_index in range(0, len(data), BATCH_SIZE):
        batch = data.iloc[start_index:start_index + BATCH_SIZE]
        prompts_for_batch = []

        for index, row in batch.iterrows():
            prompt = [f"id: {row['id']} {row['内容']}"]
            prompts_for_batch.extend(prompt)

        complete_prompt = prompt_intro + prompts_for_batch
        print(complete_prompt)
        classification = get_classification(complete_prompt)

        results = classification.split("\n")[2:]# Exclude header and separator rows.
        print(results)
        for i, row in enumerate(batch.iterrows()):
            result_file.write(results[i] + "\n")

        time.sleep(30)

print("Finished!")
