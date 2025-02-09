import argparse
import random
import time
import torch
import torch.nn as nn
import math

def custom_cross_entropy_loss(logits, targets, ignore_index=None):
    batch_size, vocab_size = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)
    if ignore_index is not None:
        mask = targets != ignore_index
    else:
        mask = torch.ones_like(targets, dtype=torch.bool)  # All True (batch_size,)
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    # consider valid target positions
    loss = -target_log_probs * mask  # Zero out ignored tokens
    # Normalize
    return loss.sum() / mask.sum()


def read_corpus(file_name,vocab,words,corpus,threshold):
    wID = len(vocab)
    count = 0
    if threshold > -1:
        with open(file_name,'rt', encoding='utf-8') as f:
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
           
                    
    with open(file_name,'rt', encoding='utf-8') as f:
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
        self.hidden = nn.Linear(dim * (window - 1), 100)
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
def batchify_data(opt):
    batch_data = []  
    num_batches = (len(opt.train) - opt.window) // opt.batchsize

    for i in range(0, num_batches * opt.batchsize, opt.batchsize):
        batch_contexts = []
        batch_targets = []
        
        for j in range(i, i + opt.batchsize):
            if j + opt.window >= len(opt.train):
                break
            context = opt.train[j:j + opt.window - 1]  
            target = opt.train[j + opt.window - 1] 
            batch_contexts.append(context)
            batch_targets.append(target)

        batch_data.append((batch_contexts, batch_targets))

    return batch_data
def train(model, opt):
    """
    Trains the Bengio model using batched data.
    """
    model.train()
    device = next(model.parameters()).device  
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_fn = nn.CrossEntropyLoss()

    batch_data = batchify_data(opt)
    num_batches = len(batch_data)
    print(f"Total Batches per Epoch: {num_batches}")

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch + 1}/{opt.epochs} starting...")
        total_loss = 0
        start_time = time.time()

        for batch_id, (contexts, targets) in enumerate(batch_data):
            contexts_tensor = torch.tensor(contexts, dtype=torch.long, device=device)
            targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

            optimizer.zero_grad()
            output = model(contexts_tensor)  # Forward pass
            loss = loss_fn(output, targets_tensor)  # Compute loss
            loss.backward()
            optimizer.step()  # Update model weights

            total_loss += loss.item()

            # Print progress every 1000 batches
            if batch_id % 1000 == 0:
                print(f"Batch {batch_id}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{opt.epochs}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

def test_model(model, opt, epoch):
    """
    Evaluates the model on the test set and calculates perplexity.
    """
    model.eval()
    device = next(model.parameters()).device  
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = (len(opt.test) - opt.window) // opt.batchsize

    with torch.no_grad():
        for i in range(0, num_batches * opt.batchsize, opt.batchsize):
            contexts = torch.tensor(
                [opt.test[j:j + opt.window - 1] for j in range(i, i + opt.batchsize)],
                dtype=torch.long,
                device=device
            )
            targets = torch.tensor(
                [opt.test[j + opt.window - 1] for j in range(i, i + opt.batchsize)],
                dtype=torch.long,
                device=device
            )

            output = model(contexts)  # Get predictions
            loss = loss_fn(output, targets)  # Compute loss
            total_loss += loss.item()  # sum total loss

    avg_loss = total_loss / num_batches  # Compute average loss
    perplexity = math.exp(avg_loss)  # Compute perplexity

    print(f"Test Loss after epoch {epoch}: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")

    return perplexity


def main():
    
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-window', type=int, default=5)   
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-batchsize', type=int, default=32)
    parser.add_argument('-lr', type=float, default=0.00005)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
                
    opt = parser.parse_args()
    opt.verbose = False    
       
    [opt.vocab,opt.words,opt.train] = read_corpus('wiki2.train.txt',[],{},[],opt.threshold)

    print(opt.train[0:100])
    # opt.train = opt.train[:100000]  # Use only 100,000 tokens for debugging


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
    with open('examples.txt','rt', encoding='utf-8') as f:
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
    device = device = torch.device("cuda")  # Force GPU usage
    model = bengio(dim=opt.d_model, 
                   window=opt.window, 
                   batchsize = opt.batchsize, 
                   vocab_size=len(opt.vocab), 
                   activation=torch.tanh).to(device)
    if opt.no_cuda == False:
        model = model.cuda()
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)    


    train(model,opt)
    test_model(model,opt,-1)
    
if __name__ == "__main__":
    main()     
