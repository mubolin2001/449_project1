import argparse
import random
import time
import torch
import torch.nn as nn

def read_corpus(file_name,vocab,words,corpus,threshold):
    
    wID = len(vocab)
    
    if threshold > -1:
        with open(file_name,'rt') as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
           
                    
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]

def encode(text,words):
        encoded = []
        tokens = text.split(' ')
        for i in range(len(tokens)):
            try:
                wID = words[tokens[i]][0]
            except:
                wID = words['<unk>'][0]
            encoded.append(wID)
        return encoded
            
class bengio(torch.nn.Module):
    def __init__(self, dim=50, window=3, batchsize = 1, vocab_size=33279, activation=torch.tanh):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.hidden = nn.Linear(dim * window, 100)
        self.output = nn.Linear(100, vocab_size)
        self.activation = activation
        # specify weights, activation functions and any 'helper' function needed for the neural net

    def forward(self, x):
        # perform a forward pass (inference) on a batch of concatenated word embeddings
        # hint: it may be more efficiwnt to pass a matrix of indices for the context, and
        # perform a look-up and concatenation of the word embeddings on the GPU.
        x = self.embed(x).view(x.shape[0], -1)
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

def train(model,opt):
    # implement code to split you corpus into batches, use a sliding window to construct contexts over
    # your batches (sub-corpora), you can manually replicate the functionality of datafeeder() to present 
    # training examples to you model, you can manually calculate the probability assigned to the target 
    # token using torch matrix operations (note: a mask to isolate the target woord in the numerator may help),
    # calculate the negative average ln(prob) over the batch and perform gradient descent.  you may want to loop
    # over the number of epochs internal to this function or externally.  it is helpful to report training
    # perplexity, percent complete and training speed as words-per-second.  it is also prudent to save
    # you model after every epoch.
    #
    # inputs to your neural network can be either word embeddings or word look-up indices
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    
    for epoch in range(opt.epochs):
        total_loss = 0
        num_batches = len(opt.train) - opt.window
        
        for i in range(0, num_batches, opt.batchsize):
            context = torch.tensor(opt.train[i:i + opt.window - 1], dtype=torch.long)
            target = torch.tensor(opt.train[i + opt.window - 1], dtype=torch.long)
            
            optimizer.zero_grad()
            output = model(context.unsqueeze(0))
            loss = loss_fn(output, target.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{opt.epochs}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    if opt.savename:
        torch.save(model.state_dict(), opt.savename + '/model_weights')
    return

def test_model(model, opt, epoch):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = len(opt.test) - opt.window
    
    with torch.no_grad():
        for i in range(0, num_batches, opt.batchsize):
            context = torch.tensor(opt.test[i:i + opt.window - 1], dtype=torch.long)
            target = torch.tensor(opt.test[i + opt.window - 1], dtype=torch.long)
            
            output = model(context.unsqueeze(0))
            loss = loss_fn(output, target.unsqueeze(0))
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Test Loss after epoch {epoch}: {avg_loss:.4f}")
    return

def main():
    
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-window', type=int, default=512)   
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
                
    opt = parser.parse_args()
    opt.verbose = False    
       
    [opt.vocab,opt.words,opt.train] = read_corpus('wiki2.train.txt',[],{},[],opt.threshold)
    print('vocab: %d train: %d' % (len(opt.vocab),len(opt.train)))
    [opt.vocab,opt.words,opt.test] = read_corpus('wiki2.test.txt',opt.vocab,opt.words,[],-1)
    print('vocab: %d test: %d' % (len(opt.vocab),len(opt.test)))
    [opt.vocab,opt.words,opt.valid] = read_corpus('wiki2.valid.txt',opt.vocab,opt.words,[],-1)
    print('vocab: %d test: %d' % (len(opt.vocab),len(opt.valid)))

    print('Train: %7d' % (len(opt.train)))
    print('Test:  %7d' % (len(opt.test)))
    print('Valid: %7d' % (len(opt.valid)))
    print('Vocab: %7d' % (len(opt.vocab)))
    print(' ')
    
    opt.examples = []
    with open('examples.txt','rt') as f:
        for line in f:
            line = line.replace('\n','')
            encoded = encode(line,opt.words)
            text = ''
            for i in range(len(encoded)):
                text = text + opt.vocab[encoded[i]] + ' '
            opt.examples.append(encoded)
            
            print('origianl: %s' % line)
            print('encoded:  %s' % text)
            print(' ')
            
    model = bengio(dim=opt.d_model, 
                   window=opt.window, 
                   batchsize = opt.batchsize, 
                   vocab_size=len(opt.vocab), 
                   activation=torch.tanh)
    if opt.no_cuda == False:
        model = model.cuda()
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)    
    
    train(model,opt)
    test_model(model,opt,-1)
    
if __name__ == "__main__":
    main()     
