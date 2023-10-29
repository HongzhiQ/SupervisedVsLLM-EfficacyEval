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

data = pd.read_csv("../../../data/cognitive distortion/cognitive_distortion_val_LLM.csv", encoding="gbk")
train_data = pd.read_csv("../../../data/cognitive distortion/cognitive_distortion_train_LLM.csv", encoding="gbk")

BATCH_SIZE = 10
prompt_intro = ["以下是12种认知歪曲的定义，请你记住以便进行判断。\n非此即彼：你用非黑即白的思维模式看待整个世界。只要你的表现有一点不完美，你就宣告彻底失败。\n以偏概全：在你看来，只要发生一件负面事件，就表示失败会接踵而来，无休无止。\n心理过滤：你单单挑出一件负面细节反复回味，最后你眼中的整个现实世界都变得黑暗无光。这就像一滴墨水染黑了一整杯水。\n否定正面思考：你拒绝正面的体验，坚持以这样或者那样的理由暗示自己“它们不算”。虽然这种消极信念有悖于现实体验，但你却以这种方式固执地坚持。\n妄下结论：你喜欢用消极的理解方式下结论，即使没有确切的事实证明也如此。\n读心术：如果发现他人的行为不尽如人意，你就认为是针对你的，对此你也懒得去查证。\n先知错误：你觉得事情只会越来越糟糕，对这一预言你深信不疑。在你看来，它就是铁板钉钉的事实。\n放大： 对于你的错误或他人的成就等方面，你往往会夸大它们的重要性。\n情绪化推理 ：你认为，只要有负面情绪，就足以证明事实确实非常糟糕，因为你这样想：“我感觉得出来，所以肯定就是真的。” \n“应该”句式：你习惯于用“我应该做这个”和“我不应该做那个”来鞭策自己，好像你需要被皮鞭抽一顿之后才能好好干活一样。“必须”和“应当”这类句式也会让人产生抵触情绪。这种句式运用于自己，带来的情绪后果就是内疚。而当你把“应该”句式强加于他人时，就会产生愤怒、沮丧甚至仇恨的情绪。\n乱贴标签：乱贴标签指的是用高度情绪化、充满感情色彩的语言来描述事物，这是一种极端的以偏概全的形式。此时，你不再描述自己的错误，而是给自己贴上消极的标签：“我是个废物。”如果有人惹恼了你，你又会给他贴上消极的标签：“他真是个讨厌鬼。”\n罪责归己：你把某个自己无法100%控制的事件的责任后果全揽到自己头上。罪责归己会引发内疚、羞愧和自卑感。\n罪责归他：你只会一味地责怪他人，而忽略了你自己的态度和行为也许也是问题的诱因。罪责归他人会引发愤怒、攻击，让问题变得越来越糟糕。\n下面是一些认知歪曲的例子，请你进行学习。\n1.非此即彼\n小王是一名大一新生，高中成绩一直名列前茅，高考时考入某知名高校。小王很努力地准备了期末考试，但考试当天身体不适导致考试失利。他说：“我连一个期末考试都考不好，我就是一个废物。”\n2.以偏概全\n小王非常害羞，他鼓起勇气向小张邀约一起吃晚饭，但小张有约，礼貌地拒绝了小王。小王便对自己说：“我永远都约不到对象，没人想和我约会，我这辈子注定孤独终老了。”\n3.心理过滤\n小王在期中考试后，很肯定100个题目中有13个答错了，并对此耿耿于怀，觉得自己肯定会被学校劝退。终于等到试卷发下来，老师说：“小王你很棒！100个题目里你答对了87个，是全班最高分。”但是小王依然觉得这次只是侥幸，而且错了13个，没什么值得自豪的。\n4.否定正面思考\n小王在生活中经常会听到朋友们由衷的赞美，总是很自然而然地想：“这只是他们表达友好的一种方式而已，并不是真心地赞美我，没人会真的喜欢我赞美我”。\n5.读心术\n小王走在路上，迎面走来一位朋友。直到两人擦身而过，都没有和小王打招呼。小王为此心情不好：“他对我视而不见，连招呼都不打，肯定是不喜欢我。”但是其实那位朋友只是想事情太入神而没有看到小王。\n6. 先知错误\n小王有次给他的朋友打电话，但过了相当长的一段时间，小王的朋友都没有给他回电话。小王由此变得恼火并心烦意乱，决定以后再也不打电话给他，也不再追究真相。小王对自己说：“如果我再打电话给他，他肯定会认为我纠缠不休。我可丢不起这个人。”\n7.放大\n小王有次考试不及格，为此小王很焦虑，脑海中的担忧挥之不去，“我以后就业/升学怎么办？工作单位和学校肯定会因为我挂科了不要我的。””\n8.情绪化推理\n小王每次拖延任务的时候都会觉得：“我又不想做这件事/学习了，我真是一个懒惰、不上进的人。”\n9. “应该”句式\n小王每到考试前总是对自己说：“我应该要好好复习了！”却因为压力过大，难以集中精力，导致复习效率很低，这又使小王认为自己没有好好复习，用“应该好好复习”来鞭策自己，使自己压力更大。\n10.乱贴标签\n小王与别人发生了争执，就想“TA真是个较真的人，我以后不要跟TA讨论了。”在之后所有的讨论中，小王都拒绝与这个人交流，即使TA的态度可能很友善。小王总是抱有一种“我是个没用的人，我什么都做不好”的想法，因此所有事情都不敢去尝试。\n11. 罪责归己\n小王答应给别人帮忙布置活动现场，却因临时有事没有去，结果活动现场的布景设施出了问题，小王觉得这个问题也有自己的错，要是自己当时去了就好了。\n12. 罪责归他\n小王正在准备第二天的考试，他的室友临时约他出去吃饭，小王不好意思拒绝，于是就答应了室友的邀约。结果小王第二天考试不太顺利，有一些知识点没有复习到。小王开始责怪室友，心里想：都怪他！就是因为他来约我吃饭才让我没有时间好好复习的。不然我肯定考的特别好\n 你是心理学专家，请你记住以上认知歪曲的定义和典型例子，并考虑下述段落中表达的心理健康状况，判断以下段落是否包含如下12种（非此即彼、以偏概全、心理过滤、否定正面思考、读心术、先知错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他）认知歪曲特征，请你用0和1标注认知歪曲特征，0表示不含有该认知歪曲特征，1表示含有该认知歪曲特征。请使用MarkDown表格形式输出分类结果。输出结果的格式如下：表头依次为id、非此即彼、以偏概全、心理过滤、否定正面思考、读心术、先知错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他。注意：请只生成MarkDown表格，其他文字不要有。接下来我将给你几个段落以及其相对应的这12种特征的专家标注结果（注意：以下标签的顺序从左到右依次为非此即彼、以偏概全、心理过滤、否定正面思考、读心术、消费者错误、放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他），请你学习下面的例子。\n"]
prempt_kaishi = ["\n请你根据学习到的专家标注例子，进行下述内容的判断:\n"]

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

        results = classification.split("\n")[2:]   # Exclude header and separator rows.
        print(results)
        for i, row in enumerate(batch.iterrows()):
            result_file.write(results[i] + "\n")

        time.sleep(30)

print("Finished!")
