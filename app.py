"""
@Author: Moussa Kalla
Date: 18/12/2024
"""
from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tensorflow as tf
import numpy as np

# Création de l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    """
    Page d'accueil
    """
    return render_template('index.html', message="Bienvenue dans votre Application")

@app.route('/hugging', methods=['GET', 'POST'])
def hugging():
    """
    Page pour tester un modèle Hugging Face
    """
    message = "Veuillez entrer votre Texte !!"
    if request.method == 'POST':
        user_input = request.form['message']
        if user_input.strip():
            # Charger le modèle GPT-2
            model_name = "gpt2"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Convertir le texte en tokens
            inputs = tokenizer(user_input, return_tensors="pt")
            # Générer du texte
            outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
            # Décoder le texte généré
            message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return render_template('hugging.html', message=message)

@app.route('/tensor', methods=['GET', 'POST'])
def tensor():
    """
    Page pour tester un modèle TensorFlow
    """
    message = "Veuillez entrer votre modèle !!"
    if request.method == 'POST':
        # Récupérer les valeurs entrées par l'utilisateur
        modele = request.form['modele']
        x_value = request.form['x']
        try:
            x_value = float(x_value)
            # Générer des données simples
            x_train = np.array([0, 1, 2, 3, 4], dtype=float)
            y_train = np.array([-1, 1, 3, 5, 7], dtype=float)  # y = 2x - 1
            # Définir un modèle simple
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1, input_shape=[1])
            ])
            # Compiler le modèle
            model.compile(optimizer='sgd', loss='mean_squared_error')
            # Entraîner le modèle
            model.fit(x_train, y_train, epochs=500, verbose=0)
            # Faire une prédiction
            prediction = model.predict(np.array([x_value]))[0][0]
            message = f"Pour x={x_value}, y prédit = {prediction:.2f}"
        except ValueError:
            message = "Veuillez entrer une valeur numérique pour x."
    return render_template('tensor.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
