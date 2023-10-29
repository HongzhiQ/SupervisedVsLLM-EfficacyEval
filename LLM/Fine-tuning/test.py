import openai
import numpy as np

# your ChatGPT API key
openai.api_key = 'API-KEY'  # Please replace with your API key.

print(openai.FineTuningJob.retrieve("Enter the unique ID of the specific fine-tuning file"))

import json
with open('../../data/cognitive distortion/cognitive_distortion_val.jsonl', 'r', encoding="utf-8") as f:
    ground_truth = []
    predict_label = []
    for line in f:
        data = json.loads(line)
        completion = openai.ChatCompletion.create(
          model="ft:gpt-3.5-turbo-0613:personal::7rOa7bik",
          messages=
          data['messages'][0:2]
        )
        print(data['messages'][2]['content'],completion.choices[0].message['content'])
        if data['messages'][2]['content'] == '低自杀风险':
            ground_truth.append(0)
        else:
            ground_truth.append(1)
        if completion.choices[0].message['content'] == '低自杀风险':
            predict_label.append(0)
        else:
            predict_label.append(1)
        ground_truth_np = np.array(ground_truth)
        np.save('ground_truth_np_hy.npy', ground_truth_np)
        predict_label_np = np.array(predict_label)
        np.save('predict_label_np_hy.npy', predict_label_np)
