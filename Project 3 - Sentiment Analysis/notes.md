**Q1. Why must TF-IDF be fitted only on training data?**

**Answer:** TF-IDF learns a vocabulary and inverse document frequency weights from the corpus. Fitting it on the full dataset exposes preprocessing to held-out data and can contaminate evaluation. Therefore, I fit it on training data and use `transform()` on validation and test data.

**Q2. Why is logistic regression a reasonable baseline for text classification?**

**Answer:** TF-IDF produces numerical feature vectors that logistic regression can use efficiently. It is relatively simple, computationally inexpensive, and often performs strongly on text classification tasks, providing a benchmark for more complex models.

**Q3. If a baseline achieves 90% accuracy and a fine-tuned transformer achieves 90.3%, how would you determine whether the improvement is meaningful?**

**Answer:** I would evaluate both models on the same untouched test set, examine precision, recall, and F1-score, and assess uncertainty using an appropriate statistical comparison or bootstrap confidence intervals. A 0.3 percentage-point difference alone does not establish a meaningful improvement.

### Q1. Why shuffle the training dataset?
Shuffling changes the order of training examples between epochs, reducing dependence on a fixed data order and potentially improving optimization. It does not directly prevent overfitting.

### Q2. What does `DataCollatorWithPadding` do?
It dynamically pads tokenized examples within a batch to a common sequence length so they can be stacked into tensors efficiently. It also batches compatible fields such as `input_ids`, `attention_mask`, and `labels`.


## DistilBERT Fine-Tuning — Interview Q&A

### Q1. What is the difference between pretraining and fine-tuning?
Pretraining teaches a model general representations using large-scale data. Fine-tuning adapts an existing pretrained model to a specific downstream task using task-specific labeled data.

### Q2. What is the purpose of an attention mask?
An attention mask identifies real tokens versus padding tokens. It prevents the model from treating padding positions as meaningful input.

### Q3. Why does CrossEntropyLoss expect raw logits?
PyTorch's CrossEntropyLoss internally combines log-softmax and negative log-likelihood. Passing raw logits is numerically stable and avoids applying softmax unnecessarily.

### Q4. What happens if optimizer.zero_grad() is omitted?
PyTorch accumulates gradients by default. Without clearing previous gradients, gradients from multiple batches accumulate and produce unintended parameter updates.

### Q5. Why save the tokenizer with a fine-tuned model?
The tokenizer contains the vocabulary and tokenization configuration required to convert new text into the token IDs expected by the model. The same tokenizer configuration should be used during inference.