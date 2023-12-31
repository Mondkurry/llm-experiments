import torch
from transformer import GPTLanguageModel  # Import the model class

# Some other stuff
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(r'D:\lara\llm-experiments\project\weight_interpolation_experiment\shakespeare\shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

# Set the seed and device
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model with the saved weights
model = GPTLanguageModel().to(device)
model.eval()
model.load_state_dict(torch.load('shakespeare/model_weights.pth', map_location=device))

# Ensure the model is in evaluation mode


# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

# Optionally, you can write to a file
# open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
