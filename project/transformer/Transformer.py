import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import color
from color import magenta, green, red

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads 
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        # Explanation: Ensures that 'embed_size' is divisible by 'heads'. This is because the embedding is cut up into chunks and fed into identical but seperate attention heads. 
        # Each head sees a reduced dimension of the embedding which is concatonated at the end to form the final full form. This was better than just one single headed attention
        # according to the "Attention is all you need" paper.
        
        self.query_weights = nn.Linear(self.head_dim, self.head_dim, bias=False) # The query needs to be head_dim x head_dim because it is multiplied by the key which is head_dim x head_dim
        self.key_weights = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_weights = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, queries, keys, values, mask, testing_mode=False): # Actual Queries, Keys and Values are passed in here, not the same as weight matrices
        
        # queries, keys, values have shape: (num_examples, seq_length, embed_size)
        
        num_examples = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # Split embedding into self.heads pieces
        # queries, keys, values have a new shape: (num examples, seq length, num heads, head dimension)
        
        queries = queries.reshape(num_examples, query_len, self.heads, self.head_dim)
        keys = keys.reshape(num_examples, key_len, self.heads, self.head_dim)
        values = values.reshape(num_examples, value_len, self.heads, self.head_dim)
        
        queries = self.query_weights(queries)
        keys = self.key_weights(keys)
        values = self.value_weights(values)
        
        # Size should be: [batch size, seq length, num heads, head dimension]
        if testing_mode: 
            if queries.shape[0] == num_examples and queries.shape[1] == query_len and queries.shape[2] == self.heads and queries.shape[3] == self.head_dim: 
                print('Size of query is', green('correct'))
            else:
                print('Size of query is', red('incorrect'))
                print(queries.shape, red('does not match'), [num_examples, query_len, self.heads, self.head_dim])
        
        # Matmul Q and K
        # queries_dot_values shape: (num examples, num heads, query_len, key_len)
        # nqhd, nkhd -> nhqk (n: number of examples, q: query length, k: key length, h: number of heads, d: head dimension)
        
        queries_dot_values = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])                                           # einsum is a more efficient way of doing matrix multiplication
        queries_dot_values = queries_dot_values / np.sqrt(self.head_dim)
        
        if mask is not None:
            queries_dot_values = queries_dot_values.masked_fill(mask == 0, float("-1e20"))                              # Masking out the padded values, use -1e20 because softmax will make it close to 0
        
        attention = torch.softmax(queries_dot_values, dim=-1)                                                           # dim=-1 means the last dimension
        out = torch.einsum("nhqk, nlhd -> nqhd", [attention, values]).reshape(num_examples, query_len, self.embed_size) # multiply attention by values and reshape to original shape with embed length
        
        # Size should be:[batch size, seq length, embed size]
        if testing_mode: 
            if out.shape[0] == num_examples and out.shape[1] == value_len and out.shape[2] == self.embed_size: 
                print('Size of output is', green('correct'))
            else:
                print('Size of output is', red('incorrect'))
                print(out.shape, red('does not match'), [num_examples, value_len, self.embed_size])

class TransformerTest():    
    def test_self_attention(self):
        # TESTING SELF ATTENTION LAYERS
        print(magenta("Testing self attention"))

        embed_size = 512
        heads = 8
        seq_length = 10
        batch_size = 4

        queries = torch.rand(batch_size, seq_length, embed_size)
        keys = torch.rand(batch_size, seq_length, embed_size)
        values = torch.rand(batch_size, seq_length, embed_size)

        self_attention = SelfAttention(embed_size, heads)
        self_attention.forward(queries, keys, values, None, testing_mode=True)
    

def main():
    tests = TransformerTest()
    tests.test_self_attention()

if __name__ == "__main__":
    main()