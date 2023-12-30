import torch
from transformer import GPTLanguageModel  # Import the model class

# Some other stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Initialize the model
model = GPTLanguageModel()
m = model.to(device)

# Load the saved weights
model.load_state_dict(torch.load('model_weights.pth'))

# Ensure the model is in evaluation mode
model.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))