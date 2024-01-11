import argparse
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
import sys
import torch

#the mutual information loss
def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  p_i_j = compute_joint(x_out, x_tf_out)
  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k).clone()  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_j[(p_j < EPS).data] = EPS
  p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                            - torch.log(p_j) \
                            - torch.log(p_i))

  loss_no_lamb = loss_no_lamb.sum()

  return loss, loss_no_lamb

def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j



#text function
def count_labels_num(path):
    labels_set, columns = set(), {}
    labels_alllll=[]
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_alllll.append(label)
            labels_set.add(label)
    return len(labels_set),labels_alllll

def count_labels(path,l):
    label_1 = 0
    label_0 = 0
    for i in range(len(l)):
        if l[i] == 0:
            label_0 = label_0 + 1
        else:
            label_1 = label_1 + 1
    return label_0, label_1


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):

            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset

def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, _,logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        '''with open("/data2/lxj/pre-train/results/confusion_matrix",'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a)+'\n')'''
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion

def text_class(path):
    text1_class_0=[]
    text1_class_1=[]
    text2_class_0 = []
    text2_class_1 = []
    text3_class_0 = []
    text3_class_1 = []

    text=read_dataset(args, path)

    textall1 = []
    textall2 = []
    textall3 = []
    for i in text:
        if i[1]==0:
            text1_class_0.append(i[0])
            text2_class_0.append(i[2])
            text3_class_0.append(i[1])
        else:
            text1_class_1.append(i[0])
            text2_class_1.append(i[2])
            text3_class_1.append(i[1])

    textall1.extend(text1_class_0)
    textall1.extend(text1_class_1)
    textall2.extend(text2_class_0)
    textall2.extend(text2_class_1)
    textall3.extend(text3_class_0)
    textall3.extend(text3_class_1)
    return textall1,textall2,textall3

#image function
def imagedata(data):
    data_0=[]
    data_1=[]
    dataall=[]
    for i in range(len(data)):
        if data[i][1]==0:

            data_0.append(data[i][0].tolist())
        else:
            data_1.append(data[i][0].tolist())
    print("0yangben:",len(data_0))
    dataall.extend(data_0)

    dataall.extend(data_1)
    return dataall

#model
#text model
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.output_layer_3 = nn.Linear(args.hidden_size, 256)


    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        logits1 = self.output_layer_3(output)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits1,logits
        else:
            return None, logits1,logits
            #return temp_output, logits

#参数
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

finetune_opts(parser)

parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                    help="Pooling type.")

parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                    help="Specify the tokenizer."
                         "Original Google BERT uses bert tokenizer on Chinese corpus."
                         "Char tokenizer segments sentences into characters."
                         "Space tokenizer segments sentences into words according to space."
                    )

parser.add_argument("--soft_targets", action='store_true',
                    help="Train model with logits.")
parser.add_argument("--soft_alpha", type=float, default=0.5,
                    help="Weight of the soft targets loss.")

args = parser.parse_args()

# Load the hyperparameters from the config file.
args = load_hyperparam(args)

set_seed(args.seed)

# Count the number of labels.
args.labels_num, ll = count_labels_num(args.train_path)

# Build tokenizer.
args.tokenizer = str2tokenizer[args.tokenizer](args)

# Build classification model.
model = Classifier(args)

# Load or initialize parameters.
load_or_initialize_parameters(args, model)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(args.device)

batch_size = args.batch_size
trainset = read_dataset(args, args.train_path)
instances_num = len(trainset)

args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

#image model
# self-attention
class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        # self.in_channels=in_channels
        self.query = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.key = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.value = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape

        # q:(batch_size,33*33=1089,channel)=(?,1089,1)
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # k:(batch_size,1,1089)
        k = self.key(input).view(batch_size, -1, height * width)
        # v:(batch_size,1,1089)
        v = self.value(input).view(batch_size, -1, height * width)
        q = q.to(args.device)
        k = k.to(args.device)
        v = v.to(args.device)
        attention_matrix = torch.bmm(q, k)  #    (batch_size,1089,1089)
        attention_matrix = self.softmax(attention_matrix).to(args.device)
        out = torch.bmm(v, attention_matrix.permute(0, 2,
                                                    1))  # (batch_size,1,1089)*(batch_size,1089,1089)=(batch_size,1,1089)
        out = out.view(*input.shape)
        self.gamma = self.gamma.to(args.device)
        return self.gamma * out + input

selfattention = SelfAttention().to(args.device)


class MyModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_directions,
                 n_class):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions

        self.feature_enginnering = nn.Sequential(
            # 1*32*32
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                stride=1,
            ),
            # 1*28*28
            nn.MaxPool2d(2, 2),
            # 1*14*14
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                stride=1,
            ),
            # 1*10*10
            nn.MaxPool2d(2, 2)
            # 1*5*5
        )

        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)  # 双向
        self.fc = nn.Linear(hidden_size * num_directions, n_class)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size)
        h0=h0.to(args.device)
        c0=c0.to(args.device)
        out = self.feature_enginnering(x)
        out = out.view(out.size(0), out.size(1), -1)
        out=out.to(args.device)
        out, (hn, cn) = self.lstm(out, (h0,
                                        c0))
        return out


net=MyModel(5*5,128,2,2,2).to(args.device)

a1, b1 = count_labels(args.train_path, ll)
a2, b2 = count_labels(args.test_path, ll)

print("train_0;", a1)
print("train_1:", b1)
print("test_0;", a2)
print("test_1:", b2)

text_src,text_seg,text_tgt=text_class(args.train_path)
src = torch.LongTensor(text_src)
tgt = torch.LongTensor(text_tgt)
seg = torch.LongTensor(text_seg)
src = src.to(args.device)
tgt = tgt.to(args.device)
seg = seg.to(args.device)

#image information
train_data_transforms=transforms.Compose(
    [
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5] ,[0.5,0.5,0.5])
    ]
)

#Read image data
train_data_i = ImageFolder('./train/',transform=train_data_transforms)
print("Number of training set samples:",len(train_data_i))

data_i=imagedata(train_data_i)
data_i=np.array((data_i))
data_i=data_i.squeeze()
data_i=torch.Tensor(data_i)
data_i=data_i.to(args.device)

#Merge data
train_data = TensorDataset(src,tgt,seg,data_i)
data =Data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,drop_last=True)

#Optimizer
optimizer, scheduler = build_optimizer(args, model)
optimizer2=torch.optim.Adam(selfattention.parameters(),lr=0.001)
optimizer3=torch.optim.Adam(net.parameters(),lr=0.001)

print("Batch size: ", batch_size)
print("The number of training instances:", instances_num)

if torch.cuda.device_count() > 1:
    print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))

    model = torch.nn.DataParallel(model, device_ids=[0])
args.model = model

total_loss, result, best_result = 0.0, 0.0, 0.0

print("Start training.")

for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
    model.train()
    selfattention.train()
    net.train()
    for i, j in enumerate(data):
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer.zero_grad()

        x, y, w, m = j
        src_batch = x
        tgt_batch = y
        seg_batch = w
        loss1, output1, output3 = model(src_batch, tgt_batch, seg_batch)

        m = m[:, :1, :, :]
        output2 = selfattention(m)
        output2 = net(output2)
        output2 = output2.squeeze()  # 16 256
        loss2, loss_no_lamb = IID_loss(output2, output1,lamb=0.1)
        loss=loss2+loss1
        loss.backward()

        optimizer2.step()
        optimizer3.step()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if (i + 1) % args.report_steps == 0:
            print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                              total_loss / args.report_steps))
            total_loss = 0.0
    result = evaluate(args, read_dataset(args, args.dev_path), True)


model.eval()
# Evaluation phase
if args.test_path is not None:
    print("Test set evaluation.")
    evaluate(args, read_dataset(args, args.test_path), True)
