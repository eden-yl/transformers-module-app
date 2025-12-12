from transformers import pipeline

# 1. Define the task and create the pipeline
classifier = pipeline("sentiment-analysis")

# 2. Use the classifier on your input
result = classifier("I've successfully installed the transformers library and started my project!")

# You can test with a negative phrase as well:
result_negative = classifier("This library seems complicated and difficult to use.")

# 3. Print the results
print("Positive Test:", result)
print("Negative Test:", result_negative)