# 🌐 Traductor Multimodal de Texto e Imágenes (EN ↔ ES)

Sistema de traducción automática neuronal **inglés ↔ español** basado en la arquitectura **Transformer** (paper *Attention Is All You Need*), con soporte **multimodal**: entrada por texto o por imagen mediante **OCR**.

## 📖 Descripción

Este proyecto implementa un pipeline completo de traducción automática neuronal (NMT) utilizando modelos preentrenados **MarianMT** de Helsinki-NLP en HuggingFace. Su propósito es evidenciar el funcionamiento del **mecanismo de atención**, la **codificación posicional**, los **embeddings** y la estructura **encoder-decoder** de un Transformer aplicados a una tarea secuencial real. Adicionalmente, integra **EasyOCR** para extraer texto de imágenes, demostrando un caso de uso multimodal y la relación directa entre la calidad del contexto de entrada y la calidad de la predicción.

## ✨ Características

Traducción bidireccional Español → Inglés e Inglés → Español, OCR multimodal con EasyOCR, aceleración automática por GPU (CUDA) con fallback a CPU, interfaz web interactiva con Gradio en pestañas de texto e imagen, modelos Transformer preentrenados de HuggingFace, notebook reproducible en Google Colab y compatibilidad con Transformers v5 mediante `AutoModelForSeq2SeqLM`.

## 🏗️ Componentes del Transformer aplicados

El sistema utiliza `AutoTokenizer` para los embeddings, codificación posicional interna de MarianMT, self-attention en el encoder, cross-attention en el decoder, feed-forward networks y layer normalization distribuidas en las capas internas del modelo.

## 🚀 Instalación

En Google Colab, abre el notebook y ejecuta las celdas en orden; la primera instala las dependencias con `!pip install -q gradio transformers sentencepiece sacremoses easyocr`. En local, clona el repositorio con `git clone https://github.com/TU_USUARIO/transformers-traductor.git`, entra en la carpeta e instala con `pip install -r requirements.txt`. Las dependencias requeridas son torch, transformers, sentencepiece, sacremoses, easyocr, gradio, pillow y numpy.

## 🎮 Uso

Ejecuta `python app.py` para abrir la interfaz Gradio con dos pestañas: traducción de texto e ingreso de imagen para OCR. También puedes usarlo desde Python importando la clase `Translator` y llamando al método `traducir("A Quiet Afternoon", "Inglés ➔ Español")`, que devuelve "Una tarde tranquila".

## 🧪 Resultados y análisis

En la prueba con la imagen *"A Quiet Afternoon"* (lectura A1 en inglés), el OCR detectó texto con ruido como `"Veftuba ingles Facu A Quiet Afternoon Toay hiau atterraon onme Me sit by the windo It is raining outside ht my roxom Wlarm..."` y la traducción resultante fue `"Veftuba ingles Facu Una tarde tranquila Toay hiau atterraon onme Me siento junto al viento Está lloviendo fuera de mi roxom Wlarm..."`. El análisis muestra que el Transformer funciona correctamente en las frases limpias (`A Quiet Afternoon → Una tarde tranquila`, `It is raining outside → Está lloviendo fuera`), confirmando que el mecanismo de cross-attention alinea fuente y destino. Los tokens con ruido OCR (`Veftuba`, `Toay`, `hiau`, `roxom`) degradan la atención y se mantienen sin traducir, validando empíricamente la relación contexto → atención → predicción. Aun con ruido severo, el Transformer produce traducción comprensible en las porciones limpias gracias a la atención global, superando las limitaciones de las arquitecturas recurrentes.

## 🧩 Tecnologías utilizadas

PyTorch como framework de Deep Learning, HuggingFace Transformers para los modelos MarianMT (encoder-decoder), EasyOCR para detección de texto en imágenes, Gradio para la interfaz web y Google Colab como entorno de ejecución del notebook.

## 📚 Referencias

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 30. Documentación de HuggingFace Transformers en https://huggingface.co/docs/transformers. Modelos Helsinki-NLP OPUS-MT en https://huggingface.co/Helsinki-NLP. EasyOCR en https://github.com/JaidedAI/EasyOCR. Gradio en https://www.gradio.app/docs.

