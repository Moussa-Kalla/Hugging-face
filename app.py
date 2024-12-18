"""
@Author: Moussa Kalla
Date: 18/12/2024
"""

from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Création de l'application Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Route principale pour enregistrer les utilisateurs.
    """
    message = "Rentrez une phrase pour le test !"

    if request.method == 'POST':
        message = request.form['message']
        # Charger le modèle GPT-2 et son tokenizer
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Convertir le texte en tokens (entrée du modèle)
        inputs = tokenizer(message, return_tensors="pt")
        # Générer du texte
        outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
        # Décoder les tokens générés en texte
        message = f"{tokenizer.decode(outputs[0], skip_special_tokens=True)}"
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run()
