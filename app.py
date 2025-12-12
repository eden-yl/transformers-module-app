from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# --- Model Loading (Happens ONCE when the app starts) ---
# We use a larger model for better translation quality in a real app,
# but 't5-small' is still good for speed.
MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def translate_text(text, source_lang, target_lang):
    """Encodes, generates, and decodes the translation."""

    # 1. Create the T5 prompt (e.g., "translate English to German: Hello")
    prompt = f"translate {source_lang} to {target_lang}: {text}"

    # 2. Encode the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # 3. Generate the output
    outputs = model.generate(input_ids, max_length=50)

    # 4. Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main translation page."""
    # Pass empty results initially
    return render_template('index.html', original_text="", translated_text="")


@app.route('/translate', methods=['POST'])
def handle_translation():
    """Handles the translation request from the web form."""

    # 1. Get data from the web form
    # request.form is how Flask accesses submitted form data
    input_text = request.form.get('input_text')
    source_language = request.form.get('source_lang')
    target_language = request.form.get('target_lang')

    # 2. Perform the translation
    translation_result = translate_text(input_text, source_language, target_language)

    # 3. Render the page again with the results
    return render_template('index.html',
                           original_text=input_text,
                           translated_text=translation_result)


if __name__ == '__main__':
    # Flask app will run on http://127.0.0.1:5000/ by default
    app.run(debug=True)