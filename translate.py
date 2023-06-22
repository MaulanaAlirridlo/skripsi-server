from googletrans import Translator
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(10), wait=wait_fixed(1))
def translate(string):
    if string:
        penterjemah = Translator()
        penterjemah.raise_Exception = True
        hasil = penterjemah.translate(string, dest='en')
        return hasil.text
    else:
        print('terjadi kesalahan')