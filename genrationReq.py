from methods import load_model, load_tokenizer, generate_text
import torch
#
# """This model got trained on the entire text and took much longer to train, and yet it fails to give meaningful results."""
#
print(torch.cuda.is_available())

model1_path = "./gptt4"
sequence1 = "generate one random recipe containing cereal. A recipe contains title, categories, servings, ingredients,  directions. When you generate '##' stop"
max_len = 1000
generate_text(model1_path, sequence1, max_len)