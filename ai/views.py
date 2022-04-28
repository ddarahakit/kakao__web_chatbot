from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import tensorflow as tf
from PIL import Image
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import json
from src.NLG import NaturalLanguageGenerator
import os
import uuid

nlg = NaturalLanguageGenerator()
nlg.run_nlg("테스트")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser(
    'C:/Users/SLINFO/Downloads/boreal-conquest-347802-6d0ab01bd01a.json')


def upload_get(request):
    return render(request, "upload.html")


def upload_post(request):
    catdog_model = tf.keras.models.load_model('saved_model/catdog_model')

    if request.method == "POST":
        upload_file = request.FILES['upload_file']
        fs = FileSystemStorage(location='media', base_url='media')
        filename = fs.save(upload_file.name, upload_file)

        image = Image.open('media/' + upload_file.name)
        resized_image = image.resize((160, 160))
        image_arr = np.array(resized_image)
        predictions = catdog_model.predict(image_arr.reshape(1, 160, 160, 3))
        result = ''
        if predictions[0][0] < 0:
            result = '고양이'
        else:
            result = '강아지'
    return render(request, "result.html", {'result': result})


def chat(request):
    return render(request, "chat.html")


@csrf_exempt
def chatbot(request):
    get_data = json.loads(request.body.decode('utf-8'))
    text = get_data['message']
    data = {
        'message': nlg.run_nlg_crawl(text)
    }

    return JsonResponse(data)


def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        sample_rate_hertz=48000,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    transcripts = []
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
        transcripts.append(result.alternatives[0].transcript)

    return transcripts


def synthesize_text(text):
    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Standard-D",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    file_uuid = str(uuid.uuid1())
    with open(os.getcwd() + "\\media\\output" + file_uuid + ".ogg", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    return file_uuid


@csrf_exempt
def audio(request):
    if request.method == "POST":
        audio_file_data = request.FILES.get('data')
        fs = FileSystemStorage(location='media', base_url='media')
        filename = fs.save("audio.ogg", audio_file_data)
        transcripts = transcribe_file(os.getcwd() + "\\media\\" + filename)
    print(transcripts[0])
    result = nlg.run_nlg_crawl(transcripts[0])
    print(result)
    file_uuid = synthesize_text(result)

    data = {
        'message': "output" + file_uuid + ".ogg"
    }

    return JsonResponse(data)



@csrf_exempt
def audio_ffmpeg(request):
    if request.method == "POST":
        audio_file_data = request.FILES.get('data')
        fs = FileSystemStorage(location='media', base_url='media')
        filename = fs.save("audio.ogg", audio_file_data)
        input_file = os.getcwd() + "\\media\\" + filename
        output_file = os.getcwd() + "\\media\\" + filename.split(".")[0] + ".wav"
        os.system("ffmpeg.exe -i " + input_file + " " + output_file + "")
        transcripts = transcribe_file(output_file)
    print(transcripts[0])
    result = nlg.run_nlg_crawl(transcripts[0])
    print(result)
    file_uuid = synthesize_text(result)

    data = {
        'message': "output" + file_uuid + ".ogg"
    }

    return JsonResponse(data)
