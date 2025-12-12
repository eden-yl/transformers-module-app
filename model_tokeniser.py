from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the name of the pre-trained model you want to use
# 't5-small' is a good choice to start, as it's smaller and faster to download.
model_name = "t5-small"

# 1. Load the Tokenizer
# AutoTokenizer is a convenient class that automatically selects the correct tokenizer
# based on the model_name you provide.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load the Model
# AutoModelForSeq2SeqLM loads the specific type of T5 model suitable for sequence-to-sequence tasks.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Prepare an input text
text = "translate English to German: The transformers library is amazing."

# 4. Encode the input text (convert text to numerical tokens)
# return_tensors='pt' specifies PyTorch tensors as the output format.
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 5. Generate the output tokens from the model
# The 'max_length' parameter prevents the generation from running indefinitely.
outputs = model.generate(input_ids, max_length=50)

# 6. Decode the output tokens back into human-readable text
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {text}")
print(f"Output: {decoded_output}")

# Example output:
# Output: Die Transformer-Bibliothek ist erstaunlich.