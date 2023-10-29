from transformers import BertModel
from model import StructuredSelfAttention
from train import train
import torch
import utils as utils
from dataget import create_data

config = utils.read_config("config.yml")
if config.GPU:
    torch.cuda.set_device(0)
print('loading data...\n')
label_num = 12

train_data = create_data("../../data/cognitive distortion/cognitive_distortion_train_LSAN.csv", batch_size=config.batch_size)
test_data = create_data("../../data/cognitive distortion/cognitive_distortion_val_LSAN.csv", batch_size=config.batch_size)
model = BertModel.from_pretrained("../../bert-base-chinese")
for param in model.parameters():
    print(param.shape)
params = [param for param in model.parameters()]
embed=params[0]
label_embed = None
print("load done")

def multilabel_classification(attention_model, train_loader, test_loader, epochs, GPU=True):
    loss = torch.nn.BCELoss()
    opt = torch.optim.Adam(attention_model.parameters(), lr=0.00005, betas=(0.9, 0.99))
    train(attention_model, train_loader, test_loader, loss, opt, epochs,GPU)

#attention model
attention_model = StructuredSelfAttention(batch_size=config.batch_size, lstm_hid_dim=config['lstm_hidden_dimension'],
                                          d_a=config["d_a"], n_classes=label_num, label_embed=label_embed,embeddings=embed)

if config.use_cuda:
    attention_model.cuda()

multilabel_classification(attention_model, train_data, test_data, epochs=config["epochs"])



