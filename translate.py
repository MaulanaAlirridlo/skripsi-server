from googletrans import Translator

def translate(string):
  if string:
    penterjemah = Translator()
    penterjemah.raise_Exception = True
    hasil = penterjemah.translate(string, dest='en')
    return hasil.text
  else:
    print('terjadi kesalahan')