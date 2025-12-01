from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Variables globales para manejar el estado de enseñanza
teaching_mode = False  # Indica si el chatbot está en modo de enseñanza
new_symptom = ""  # Almacena el nuevo síntoma que se está enseñando

# Cargar los archivos generados por el chatbot
lemmatizer = WordNetLemmatizer()  # Inicializa el lematizador de palabras
intents = json.load(open('intents.json', encoding='utf-8'))  # Carga las intenciones desde un archivo JSON
words = pickle.load(open('words.pkl', 'rb'))  # Carga las palabras desde un archivo pickle
classes = pickle.load(open('classes.pkl', 'rb'))  # Carga las clases desde un archivo pickle
model = load_model('chatbot_model.h5')  # Carga el modelo de chatbot entrenado

# Funciones del chatbot
def clean_up_sentence(sentence):
    # Tokeniza la oración y aplica lematización a cada palabra
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # Convierte una oración en un vector binario basado en las palabras conocidas
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  # Inicializa el vector de palabras con ceros
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Marca la presencia de la palabra en el vector
    return np.array(bag)

def predict_class(sentence):
    # Predice la clase de una oración usando el modelo entrenado
    bow = bag_of_words(sentence)  # Obtiene el vector binario de la oración
    res = model.predict(np.array([bow]))[0]  # Realiza la predicción usando el modelo
    max_index = np.argmax(res)  # Encuentra el índice de la mayor confianza
    category = classes[max_index]  # Obtiene la clase correspondiente al índice
    confidence = res[max_index]  # Obtiene la confianza de la predicción
    return category, confidence

def save_new_symptom(symptom, response):
    # Guarda un nuevo síntoma y su respuesta en el archivo de intenciones
    new_tag = f"new_symptom_{len(intents['intents']) + 1}"  # Genera una nueva etiqueta para el síntoma
    new_intent = {
        "tag": new_tag,
        "patterns": [symptom],  # Agrega el síntoma a los patrones
        "responses": [response]  # Agrega la respuesta correspondiente
    }
    intents['intents'].append(new_intent)  # Añade el nuevo intento a la lista de intenciones
    # Guarda las intenciones actualizadas en el archivo JSON
    with open('intents.json', 'w', encoding='utf-8') as file:
        json.dump(intents, file, ensure_ascii=False, indent=4)

def get_response(tag, intents_json, message, confidence, confidence_threshold=0.7):
    # Obtiene una respuesta basada en la etiqueta de intención y la confianza
    global teaching_mode, new_symptom

    list_of_intents = intents_json['intents']  # Obtiene la lista de intenciones
    result = ""

    if teaching_mode:
        if not new_symptom:
            new_symptom = message  # Almacena el nuevo síntoma para la enseñanza
            result = "Ahora escribe la respuesta."
        else:
            save_new_symptom(new_symptom, message)  # Guarda el nuevo síntoma y su respuesta
            new_symptom = ""
            teaching_mode = False
            result = "Tu información se guardó correctamente."
        return result

    if confidence < confidence_threshold:
        if message.lower() in ["te quiero enseñar", "nuevo"]:
            teaching_mode = True  # Activa el modo de enseñanza
            result = "Okay, escribe el síntoma."
        else:
            # Si la confianza es baja y no se encuentra en modo enseñanza, buscar en los patrones conocidos
            for intent in intents_json['intents']:
                if message.lower() in [pattern.lower() for pattern in intent['patterns']]:
                    tag = intent['tag']
                    result = random.choice(intent['responses'])  # Selecciona una respuesta aleatoria
                    return result
            result = f"No entiendo... '{message}'. Si quieres me lo puedes enseñar, diciendo nuevo."
    else:
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i['responses'])  # Selecciona una respuesta aleatoria basada en la etiqueta
                break
    
    return result

def save_interaction_to_log(user_input, bot_response):
    # Guarda la interacción del usuario y la respuesta del bot en un archivo de registro
    with open('interaction_log.txt', 'a') as f:
        f.write(f"Usuario: {user_input} - Bot: {bot_response}\n")

# Rutas de la aplicación web
@app.route('/')
def index():
    # Renderiza la página principal
    return render_template('index.html')

@app.route('/get_response', methods=["POST"])
def get_bot_response():
    # Obtiene la respuesta del bot para la entrada del usuario
    user_input = request.get_json().get("message")  # Obtiene el mensaje del usuario del JSON de la solicitud
    bot_response = respuesta(user_input)  # Llama a la función `respuesta` para obtener la respuesta del bot
    save_interaction_to_log(user_input, bot_response)  # Guarda la interacción en el registro
    return jsonify({"response": bot_response})  # Devuelve la respuesta en formato JSON

def respuesta(message):
    # Maneja la respuesta del bot al mensaje del usuario
    category, confidence = predict_class(message)  # Predice la clase del mensaje y la confianza
    res = get_response(category, intents, message, confidence)  # Obtiene la respuesta basada en la predicción
    return res

@app.route('/survey')  # Ruta para mostrar el formulario de encuesta
def survey_form():
    return render_template('survey.html')  # Renderiza la página del formulario de encuesta

@app.route('/submit_survey', methods=["POST"])
def submit_survey():
    # Maneja el envío de los datos de la encuesta
    fullName = request.form.get('fullName')  # Obtiene el nombre completo del formulario
    satisfaction = request.form.get('satisfaction')  # Obtiene la satisfacción del formulario
    feedback = request.form.get('feedback')  # Obtiene el comentario del formulario
    
    # Guardar los datos en el archivo de registro
    save_survey_data(fullName, satisfaction, feedback)

    # Devolver una respuesta en forma de JavaScript para mostrar una alerta y redirigir al usuario
    return """
    <script>
        alert("Gracias por tu comentario.");
        window.location.href = "/";
    </script>
    """

def save_survey_data(fullName, satisfaction, feedback):
    # Guarda los datos de la encuesta en un archivo de registro
    with open('survey_log.txt', 'a') as f:
        f.write(f"Nombre: {fullName}\n")
        f.write(f"Satisfacción: {satisfaction}\n")
        f.write(f"Comentario: {feedback}\n\n")

# Ejecutar la aplicación Flask en modo de depuración
if __name__ == '__main__':
    app.run(debug=True)
