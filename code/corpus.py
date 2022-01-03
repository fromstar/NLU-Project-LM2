import os
import torch

# Dataset files directory
data = "input"

class Corpus(object):
    def __init__(self, device):
        self.dictionary = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.len_dict = len(self.dictionary)
        self.train = self.to_token(os.path.join(data, 'ptb.train.txt'), device)
        self.test = self.to_token(os.path.join(data, 'ptb.test.txt'), device)
        self.valid = self.to_token(os.path.join(data, 'ptb.valid.txt'), device)

# Fill the dictionary and return an array with the corresponding key of the words read
    def to_token(self, path, device):
        if os.path.exists(path):  # check if the file I need to read exists
            with open(path) as txt:
                key = self.len_dict
                sentences = []
                values = list(self.dictionary.values())

                for line in txt:
                    tmp = []
                    # line = line.strip()
                    words = ['<bos>'] + line.split() + ['<eos>']

                    # scroll through the words of a sentence
                    for word in words:
                        # if the world is not in the dictionary I add it.
                        if word not in values:
                            # the length of the dictionary coincides with the index of insertion in it
                            self.dictionary[key] = word
                            tmp.append(key)
                            key += 1
                            values.append(word)
                        else:
                            tmp.append(values.index(word))
                    sentences.append(torch.LongTensor(tmp).to(device))

            print("Sentences loaded")
            self.len_dict = len(self.dictionary)
            return sentences
        else:
            raise ValueError(path +" doesn't exist.")

    def print_dic(self):
        print(self.dictionary)
        print("Number of tokens: " + str(len(self.dictionary)))