from methods import read_documents_from_directory, train
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
# train_file_path = "/content/drive/MyDrive/ColabNotebooks/data/chatbot_docs/combined_text/full_text/train.txt"
train_file_path = "medium.txt"
model_name = 'gpt2'
# output_dir = '/content/drive/MyDrive/ColabNotebooks/models/chat_models/custom_full_text'
output_dir = './gptt4'
overwrite_output_dir = True
per_device_train_batch_size = 8
num_train_epochs = 50.0
save_steps = 50000

train_directory = "./"
# text_data = read_documents_from_directory(train_directory)
# text_data = re.sub(r'\n+', '\n', text_data).strip()


train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)
