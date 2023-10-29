from pytorch_pretrained_bert import BertForSequenceClassification,BertAdam,BertModel
import codecs,json,csv,torch,os,logging
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from sklearn import metrics
import numpy as np
from torch.nn import BCELoss,BCEWithLogitsLoss,MSELoss,Module
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1)
def calculate_evaluation(prediction,true_label,type):
    recall_list=[]
    precision_list=[]
    f1_list=[]
    for i in range(0,len(true_label)):
        recall=metrics.recall_score(true_label[i],prediction[i],average=type)
        recall_list.append(recall)
        precision=metrics.precision_score(true_label[i],prediction[i],average=type)
        precision_list.append(precision)
        f1=metrics.f1_score(true_label[i],prediction[i],average=type)
        f1_list.append(f1)
    recall_list=np.array(recall_list)
    precision_list=np.array(precision_list)
    f1_list=np.array(f1_list)
    return np.mean(recall_list),np.mean(precision_list),np.mean(f1_list)


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyPro(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'cognitive_distortion_train_BERT.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'cognitive_distortion_val_BERT.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.tsv')), 'test')

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, dicts, set_type):
        examples = []
        for (i, infor) in enumerate(dicts):
            guid = "%s-%s" % (set_type, i)

            text_a = infor[12]
            label = infor[0:12]

            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        label_id = []
        for item in example.label:
            label_id.append(label_map[item])
        if ex_index < 5 and show_exp:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features



def train(model, processor, data_dir, max_seq_length,train_batch_size,label_list, tokenizer, device,optimizer,threshold=0.5):
    criterion = MSELoss()
    train_examples = processor.get_train_examples(data_dir)
    train_features = convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer,show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    predict = np.zeros((0, 12), dtype=np.int32)
    gt = np.zeros((0, 12), dtype=np.int32)
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        model.zero_grad()
        optimizer.zero_grad()
        pred = model(input_ids, segment_ids, input_mask)

        loss = criterion(pred,label_ids.float())

        loss.backward()
        optimizer.step()

        logits = np.multiply(pred.cpu().detach().numpy() >= threshold, 1)
        predict = np.concatenate((predict, logits))
        gt = np.concatenate((gt, label_ids.cpu().numpy()))

    recall, precision, f1 = calculate_evaluation(predict, gt, type='macro')
    return precision, recall, f1

def val(model, processor, data_dir, max_seq_length,eval_batch_size,label_list,tokenizer,device,threshold=0.5):
    eval_examples = processor.get_dev_examples(data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list,max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    predict = np.zeros((0, 12), dtype=np.int32)
    gt = np.zeros((0, 12), dtype=np.int32)
    model.eval()
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            pred = model(input_ids, segment_ids, input_mask)
            logits = np.multiply(pred.cpu().numpy()>=threshold,1)
            predict = np.concatenate((predict, logits))
            gt = np.concatenate((gt, label_ids.cpu().numpy()))


    recall,precision,f1=calculate_evaluation(predict, gt, type='macro')

    return precision, recall, f1

class BertForMultiLabelSequenceClassification(Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config, num_labels=12,drop=0.5):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=config)
        self.dropout = torch.nn.Dropout(drop)
        self.fc1 = torch.nn.Linear(768,768)
        self.fc2 = torch.nn.Linear(768,768)
        self.bn1 = torch.nn.BatchNorm1d(num_features=768)
        self.bn2 = torch.nn.BatchNorm1d(num_features=768)
        self.classifier = torch.nn.Linear(768, num_labels)
        #self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.relu(self.fc1(pooled_output))
        pooled_output = self.bn1(pooled_output)
        pooled_output = F.relu(self.fc2(pooled_output))
        pooled_output = self.bn2(pooled_output)
        logits = self.classifier(pooled_output)
        y = torch.sigmoid(logits)
        return y


    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

def main():

    max_seq_length = 256
    train_batch_size = 8
    val_batch_size = 8
    num_train_epochs = 100
    processor = MyPro()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForMultiLabelSequenceClassification('bert-base-chinese').to(device)
    print(model)
    optimizer = BertAdam(model.parameters(), lr=5e-6, eps=1e-8, weight_decay=1e-3)
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    dir = '../../data/cognitive distortion'
    for epoch in range(num_train_epochs):
        train(model, processor, dir, max_seq_length, train_batch_size, label_list, tokenizer, device, optimizer)
        precision, recall, f1 = val(model, processor, dir, max_seq_length, val_batch_size, label_list,
                                    tokenizer, device)
        print('epoch:', epoch, '  F1:', f1, '  recall:', recall, '  precision:', precision)
        if best_f1 < f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
    print('best_f1:', best_f1, '  best_recall:', best_recall, '  best_precision:', best_precision)


'''
model = BertForMultiLabelSequenceClassification('bert-base-chinese').to(device)
print(model)
data_dir = 'input'
vocab_dir = 'bert-base-chinese'
max_seq_length = 18
train_batch_size = 1
val_batch_size = 1
num_train_epochs = 5
processor = MyPro()
label_list = processor.get_labels()
tokenizer = BertTokenizer.from_pretrained(os.path.join(vocab_dir, 'vocab.txt'))
dir = 'renzhiwaiqu/1'
val(model, processor, dir, max_seq_length, val_batch_size, label_list,tokenizer, device)
'''
main()