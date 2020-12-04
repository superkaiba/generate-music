from flask import Flask, render_template, request, send_file, send_from_directory, safe_join, abort
from model import MusicGenerator

app = Flask(__name__)
most_recent=None

# Base endpoint to perform prediction.
@app.route('/', methods=['POST'])
def answer_query():
    print(request.form)
    if 'sequence_length' in request.form:
        music_generator = MusicGenerator()
        fname=music_generator.generate(
            sequence_length=int(request.form['sequence_length']),
            weights_path="webapp-data/weights.hdf5",
            generation_data_path=f"webapp-data/{request.form['composer']}-data.npy",
            midi_output_dir="static/midi",
            wav_output_dir="static/audio",
            )
        return render_template("index.html", audiofile=fname)

    elif 'download_midi' in request.form:
        audiofile = "static/midi/" + request.form['current_audio'][:-4].split("/")[-1] + ".midi"
        return send_file(audiofile, attachment_filename=audiofile[12:], as_attachment=True)

    elif 'download_wav' in request.form:
        audiofile = request.form['current_audio']
        return send_file(audiofile, attachment_filename=audiofile[12:], as_attachment=True)

@app.route('/', methods=['GET'])
def load():
    return render_template('index.html', prediction=None, generated_text=None, audiofile=None)


@app.route('/predict/image', methods=['POST'])
def make_image_prediction():
    prediction = "tomato"
    print(prediction)
    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True)
