import pandas as pd
from deep_translator import GoogleTranslator as gt
import re
import numpy as np
import edgeTTS
import requests
import json


def add_lines(b, data3, lim_l):
    for i3 in range(b.shape[0]):
        a = np.array([' ' * lim_l for _ in range(30)])
        a[:b[i3].shape[0]] = b[i3]
        data3 = data3.append(pd.Series(a), ignore_index=True)
    data3.iloc[-1, b[i3].shape[0]:] = np.nan
    a = np.array([np.nan for _ in range(30)])
    data3 = data3.append(pd.Series(a), ignore_index=True)
    return data3


async def text2learn(text:  'str: input text',
                     path:  'path to write output .xlcx file and tts .mp3'=None,
                     tts=False,
                     lim:   'max words in row'=10,
                     lim_l: 'max letters in translated row' = 100
                     ) ->   'pd.DataFrame with word/phonetic/translation per sell':
    """Convert english text to tab with per words phonetic and translation"""


    text4ipa = re.sub(r'(\n)+', ' ', text.lower())
    text4ipa = re.sub(r'[?!.]/w', ' ', text4ipa)
    text4ipa = re.sub(r'( )+', ' ', text4ipa)

    # phonetic
    phon = requests.get("https://tophonetics-api.ajlee.repl.co/api",
                        data={"text": text4ipa}).text
    ipa = phon.split(' ')

    # words translate
    text2 = text4ipa.replace(' ', ' \n ')
    text2 = re.sub(r'/W+', '', text2).replace(' the ', ' ').replace(' of ', ' ')
    ru_w = gt(source='auto', target='ru').translate(text2).lower()
    ru_w2 = ru_w.split('\n')

    # english
    text4 = re.sub(r'( )+', ' ', text)
    text4 = re.sub(r' $', '', text4)
    en = re.sub(r'(\n)+', ' ', text4).split(' ')

    # check words/phonetic/translation has the same length
    assert (len(en) == len(ipa) == len(ru_w2))

    # print(en, ipa, ru_w2)

    # sent translate
    res = []
    for i, word in enumerate(en):
        if len(re.findall(r'[.?!]$', word)) > 0:
            res.append(i)
    text4rut = re.sub(r'(\n)+', '', text)
    text4rut = re.sub(r'( )+', ' ', text4rut)
    ru_t = gt(source='auto', target='ru').translate(re.sub(r'([.?!]+){1}', r'\1\n', text4rut)).split('\n')


    # create output DataFrame
    data3 = pd.DataFrame()
    data = pd.DataFrame(data={'en': en, 'ipa': ipa, 'ru_w': ru_w2})

    i_old = 0
    for i in range(1, data.shape[0] + 1):
        if i % lim == 0:
            b = data[i_old:i].transpose().to_numpy()
            data3 = add_lines(b, data3, lim_l)
            i_old = i
        if i - 1 in res:
            b = data[i_old:i].transpose().to_numpy()
            data3 = add_lines(b, data3, lim_l)
            i_old = i
            b = np.array([[ru_t[res.index(i - 1)][i2 * lim_l:(i2 + 1) * lim_l]] for i2 in
                          range(len(ru_t[res.index(i - 1)]) // lim_l + 1)])
            data3 = add_lines(b, data3, lim_l)

    # write output
    if path is not None:
        data3.to_excel(path + text[:20] +'.xlsx')

    # create text-to-speech
    if tts:
        await text2speech(text, path, split=True)
    return data3


async def text2speech(text, path, voice=None, rate='-50%', split=False):

    path += 'tts/'
    if split:
        text = re.split(r'[.!?]', text)

    if voice is None:
        trusted_client_token = '6A5AA1D4EAFF4E9FB37E23D68491D6F4'
        # wssUrl = 'wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=' + trusted_client_token
        voiceList = ('https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list?trustedclienttoken='
                     + trusted_client_token)
        data = json.loads(requests.get(voiceList).text)
        voices = [x['Name'] for x in data if 'Natural' in x['FriendlyName'] and x['Locale'] == 'en-GB']
        voice = voices[0]

    # async def main():
    communicate = edgeTTS.Communicate()
    for sent_n, sent in enumerate(text):
        with open(path + f'{str(100001 + sent_n)[1:]}. {(sent[:20]).strip()}.mp3', 'wb') as fp:
            async for i in communicate.run(sent, voice=voice, rate=rate):
                if i[2] is not None:
                    fp.write(i[2])
    #             time.sleep(1)

    # if __name__ == "__main__":
    #     await main()
    return


