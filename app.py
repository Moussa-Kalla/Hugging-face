"""
@Author: Moussa Kalla
Date: 18/12/2024
"""

from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Création de l'application Flask
app = Flask(__name__)

# Charger le modèle GPT-2 et son tokenizer une seule fois au démarrage
MODEL_NAME = "gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Route principale pour traiter les requêtes utilisateur.
    """
    message = "Rentrez une phrase pour le test !"

    if request.method == 'POST':
        # Récupérer l'entrée utilisateur
        user_input = request.form['message']
        if user_input.strip():  # Vérifier que le champ n'est pas vide
            # Convertir l'entrée en tokens
            inputs = tokenizer(user_input, return_tensors="pt")
            # Générer du texte
            outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
            # Décoder les tokens générés
            message = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            message = "Veuillez entrer un texte valide."

    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)  # Activer le mode débogage pour un meilleur suivi
