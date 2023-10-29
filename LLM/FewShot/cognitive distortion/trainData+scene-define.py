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

def get_examples(data, num_examples=2):
    categories = ["非此即彼", "以偏概全", "心理过滤", "否定正面思考", "读心术", "先知错误", "放大", "情绪化推理", "应该式", "乱贴标签", "罪责归己", "罪责归他"]
    examples = []

    for category in categories:
        positive_samples = data[data[category] == 1].sample(num_examples)
        for _, row in positive_samples.iterrows():
            examples.append(f"{row['内容']},{row['非此即彼']},{row['以偏概全']},{row['心理过滤']},{row['否定正面思考']},{row['读心术']},{row['先知错误']},{row['放大']},{row['情绪化推理']},{row['应该式']},{row['乱贴标签']},{row['罪责归己']},{row['罪责归他']}")

    return examples

data = pd.read_csv("../../data/1_data_val_biaotou_cleantext.csv", encoding="gbk")
train_data = pd.read_csv("../../data/1_data_train_biaotou_cleantext.csv", encoding="gbk")

BATCH_SIZE = 10
prompt_intro = ["下方内容分别显示了用户的ID和其在社交媒体上的发文内容。请你进行多分类任务，判断帖子是否包含如下12种（非此即彼、以偏概全、心理过滤、否定正面思考、读心术、先知错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他）认知歪曲特征，请你用0和1标注认知歪曲特征，0表示不含有该认知歪曲特征，1表示含有该认知歪曲特征。请使用MarkDown表格形式输出分类结果。输出结果的格式如下：表头依次为id、非此即彼、以偏概全、心理过滤、否定正面思考、读心术、先知错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他。注意：请只生成MarkDown表格，其他文字不要有。接下来我将给你社交媒体中用户的发言内容以及这段发言相对应的这12种特征的专家标注结果（注意：以下标签的顺序从左到右依次为非此即彼、以偏概全、心理过滤、否定正面思考、读心术、消费者错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他），请你学习下面的例子。"]
prempt_kaishi = ["请你根据学习到的专家标注例子，进行下述内容的判断"]

#Replace here with the path to your result file.
with open('result.txt', 'a') as result_file:
    table_header = "| id | 非此即彼 | 以偏概全 | 心理过滤 | 否定正面思考 | 读心术 | 先知错误 | 放大 | 情绪化推理 | 应该式 | 乱贴标签 | 罪责归己 | 罪责归他 |"
    table_divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    result_file.write(table_header + "\n")
    result_file.write(table_divider + "\n")

    for start_index in range(0, len(data), BATCH_SIZE):
        batch = data.iloc[start_index:start_index + BATCH_SIZE]
        prompts_for_batch = []

        for index, row in batch.iterrows():
            prompt = [f"id: {row['id']} {row['内容']}"]
            prompts_for_batch.extend(prompt)

        example_prompts = get_examples(train_data)
        complete_prompt = prompt_intro + example_prompts + prempt_kaishi + prompts_for_batch
        print(complete_prompt)
        classification = get_classification(complete_prompt)

        results = classification.split("\n")[2:] # Exclude header and separator rows.
        print(results)
        for i, row in enumerate(batch.iterrows()):
            result_file.write(results[i] + "\n")

        time.sleep(30)

print("Finished!")
