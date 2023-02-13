import main as gr

def magic_mix(input_img):
    # TODO : Implement magic_mix
    output_img = None
    return output_img

demo = gr.Inferface(magic_mix, gr.Image(shape=(None, None, 3)), "image", live=True, capture_session=True)
demo.launch()
