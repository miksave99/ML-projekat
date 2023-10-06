import csv #uvoz biblioteke za rad sa csv fajlovima
import re #uvoz biblioteke za proveru reči

def procisti_tekst(ulaz, izlaz):
    with open(ulaz, 'r', newline='') as ulaz, open(izlaz, 'w', newline='') as izlaz:
        citac = csv.reader(ulaz)
        pisac = csv.writer(izlaz)

        for red in citac:
            sreden_red = [procisti_tekst_celija(celija) for celija in red]
            pisac.writerow(sreden_red)

def procisti_tekst_celija(celija):
    ciscenje = re.sub(r'[^\w\s]', '', celija)  # Ukloni specijalne karaktere
    return ciscenje

# Formiranje očišćenog fajla
ulazna_datoteka = 'tweet_emotions.csv'
izlazna_datoteka ='tweet_emotions1.csv'

procisti_tekst(ulazna_datoteka, izlazna_datoteka)