from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-small"

# 1. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load the Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Prepare an input text (T5 requires the task prefix, e.g., "translate English to German:")
text = "translate English to German: The transformers library is amazing."

# 4. Encode the input text
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 5. Generate the output tokens from the model
outputs = model.generate(input_ids, max_length=50)

# 6. Decode the output tokens back into human-readable text
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {text}")
print(f"Output: {decoded_output}")