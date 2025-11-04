import gradio as gr
from fastai.vision.all import *

# Loading the model we trained
learn = load_learner('one_piece.pkl', pickle_module = pickle)

# Prediction function
def predict_character(img):
    pred_class, pred_idx, probs = learn.predict(img)

    return {learn.dls.vocab[i]: float(probs[i])
            for i in range(len(learn.dls.vocab))}

# Gradio app interface
demo = gr.Interface(
    fn = predict_character,
    inputs = gr.Image(type = 'pil', label = 'Upload Character Image'),
    outputs = gr.Label(num_top_classes = 10, label = 'Predictions'),
    title = 'One Piece Straw Hat Recognition',
    description = 'Upload any straw hats members image except franky and jinbe',
    examples = None,
    theme = gr.themes.Soft()
)

# Launch our app
demo.launch()