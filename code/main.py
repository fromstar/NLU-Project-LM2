import torch
import math
import time
from torch.nn.utils.rnn import pad_sequence
from corpus import Corpus
from model import RNN
import torch.nn.functional as F

# To change the number of gru layers, the embedding and hidden size open the model.py file

device = 'cuda:0'
learning_rate = 0.001
clip = 0.35
epochs = 25
batch_size = 64
eval_batch_size = 1
dropout = 0.5
interval = 50


def get_batch(source, i, batch_size):
    data = []
    target = []
    size = 0
    for sentence in source[batch_size * i: batch_size * (i+1)]:
        data.append(sentence[:-1])
        target.append(sentence[1:])
        size += len(sentence[:-1])

    # Fill the sentences with the pad tag in order to make them the same length
    # The key for the pad tag is 0.
    data = pad_sequence(data, padding_value=0)
    target = pad_sequence(target, padding_value=0)

    return data, target, size


def train(model, train, opt, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    total_size = 0

    for batch_idx in range(0, len(train) // batch_size):
        data, target, size = get_batch(train, batch_idx, batch_size)
        output, hidden = model(data, hidden)
        # Dropout to recurrent element
        # output = nn.Dropout(dropout)(output)

        loss = F.nll_loss(
            output,
            target.view(-1),
            reduction='sum',
            ignore_index=0,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        total_loss += loss.item()

        total_size += size
        if batch_idx % interval == 0 and batch_idx > 0:
            cur_loss = total_loss / total_size
            ppl = round(math.exp(cur_loss), 2)
            elapsed = time.time() - start_time
            print('epoch: ', epoch, ' | batches: ', batch_idx+1, '/', (len(train) // batch_size), ' | learning_rate: ', learning_rate,
                  '| ms/batch: ', round(elapsed * 1000 / interval, 2), ' | loss: ', round(cur_loss, 3), ' | perplexity: ', ppl)
            total_loss = 0
            start_time = time.time()
            total_size = 0

        for i in range(len(hidden)):
            hidden[i] = hidden[i].detach()


def evaluate(data_source, model):
    model.eval()
    total_loss = 0
    total_size = 0
    with torch.no_grad():
        for i in range(0, len(data_source)//eval_batch_size):
            hidden = model.init_hidden(eval_batch_size)
            data, target, size = get_batch(data_source, i, eval_batch_size)
            output, hidden = model(data, hidden)
            for i in range(len(hidden)):
                hidden[i] = hidden[i].detach()

            total_loss += F.nll_loss(
                output,
                target.view(-1),
                reduction='sum',
                ignore_index=0,
            )
            total_size += size

    return total_loss / total_size


def main():
    save = 'model_test.pt'
    torch.manual_seed(1111)
    corpus = Corpus(device)
    print("len ditc: ", corpus.len_dict)
    model = RNN(corpus.len_dict, dropout).to(device)
    print("Dictionary's length: ", corpus.len_dict, " words.")

    train_data = corpus.train
    test_data = corpus.test
    val_data = corpus.valid

    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    # opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_val_loss = None

    try:
        for epoch in range(0, epochs):
            epoch_start_time = time.time()
            train(model, train_data, opt, epoch)
            val_loss = evaluate(val_data, model)
            ppl = round(math.exp(val_loss), 2)
            print(
                "-----------------------------------------------------------------------------------------")
            print('end epoch: ', epoch, '| time: ', round((time.time() - epoch_start_time), 2), 's | valid loss: ', round(val_loss.item(), 3),
                  '| valid ppl: ', ppl)
            print(
                "-----------------------------------------------------------------------------------------\n")

            if not best_val_loss or val_loss < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('Exiting from training early')

    with open(save, 'rb') as f:
        model = torch.load(f)

    test_loss = evaluate(test_data, model)
    print("-----------------------------------------------------------------------------------------")
    print('End of training\ntest loss: ', round(
        test_loss.item(), 3), '\ntest ppl: ', math.exp(test_loss))


main()
