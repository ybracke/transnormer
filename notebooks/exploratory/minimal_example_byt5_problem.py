# %% [markdown]
# # A minimal example of my problems with training byT5
# 


# %%
import torch
import transformers
import datasets              

# %% [markdown]
# ## Helper Functions

# %%
def models_identical(model1, model2) -> bool:
    unequal_states = []
    for (name_mo, params_mo), (name_mn, params_mn) in zip(model1.state_dict().items(), model2.state_dict().items()):
        assert name_mo == name_mn
        if not torch.equal(params_mo, params_mn):
            unequal_states.append(name_mo)
    return len(unequal_states) == 0


# %% [markdown]
# ## Set-up

# %%
checkpoint_name = "google/byt5-small"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint_name).to(device)
model_orig = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint_name).to(device)

# %%
train_data = {"input_ids" : torch.tensor([[105, 114, 114, 35, 101, 100, 117, 0, 0, 0]]), 
              "attention_mask" : torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]) ,
              "labels" : torch.tensor([[117, 100, 101, 35, 114, 114, 105, -100, -100, -100]])}
# train_data = {k: v.to(device) for k, v in train_data.items()} # muss das überhaupt?
print(train_data)
# convert to a Dataset object
ds_train = datasets.Dataset.from_dict(train_data)
print(ds_train[0])
# evaluation data is just a copy of train_data
ds_eval = ds_train

# %%
epochs = 200

# %%
# mininal training arguments
training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="test",
    predict_with_generate=True,
    evaluation_strategy = "steps",
    # fp16=True, ####### Hier spielt die Musik 
    eval_steps=100,
    num_train_epochs=epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    )

# %%
# initialize the trainer
trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
)


# %% [markdown]
# # A. With `Trainer.train()`

# %%
trainer.train()

# %%
# Check whether models are identical
print("Untrained model and trained model are the same:", 
    models_identical(model_orig, model)
    )

print("Untrained model and trained model are the same:", 
    models_identical(model_orig, trainer.model)
    )

# %%
print("Model and trainer.model are the same:", 
    models_identical(model, trainer.model)
    )

# %%
print(trainer.state.log_history)

# %% [markdown]
# ## A. Application

# %%
for batch in trainer.get_train_dataloader():
    break
batch_without_labels = {k: v.to(device) for k, v in batch.items() if k != "labels"}
print(batch_without_labels)
print(trainer.model.generate(**batch_without_labels, num_beams=2, early_stopping=True, max_length=10)[0])
print(model.generate(**batch_without_labels, num_beams=2, early_stopping=True, max_length=10)[0])

# %% [markdown]
# fp16=False : tensor([  0, 117, 100, 101,  35, 114, 114, 105, 105, 114], device='cuda:0')
# fp16=True : tensor([0, 0, 1], device='cuda:0')

# %% [markdown]
# # B. pytorch-style loop but with some trainer stuff 

# %%
# create an optimizer
trainer.create_optimizer()

# get the first (and only) batch
for batch in trainer.get_train_dataloader():
    break
batch = {k: v.to(device) for k, v in batch.items()}

# loop over batches
for i in range(epochs):
    # get loss
    outputs = trainer.model(**batch)
    loss = outputs.loss
    # gradients and backprop
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

# %%
# Check whether models are identical
print("Untrained model and trained model are the same: ", 
      models_identical(model_orig, model)
      )

print("Untrained model and trained model are the same: ", 
      models_identical(model_orig, trainer.model)
      )

