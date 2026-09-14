# Formato CSV 72 V

Unico tracciato accettato: intestazione `IDMessage;Battery;Value;W/h;amps`, sei righe `ID;B0;valore;;` fino a B5, poi `;;;energia;ampere`.
Le batterie restano in verticale nella terza colonna; energia e corrente occupano la quarta e la quinta colonna. L'etichetta W/h resta quella emessa dal firmware, senza conversioni di unita'.

Il vecchio formato con righe watts e amps non e' supportato. I vecchi CSV devono essere rigenerati prima del training. I file acquisiti esistenti non sono stati modificati.

Conteggio, training e inferenza condividono il parser battery_csv.h: x[0] = ampere, x[1] = energia, uscite = B0...B5. Il parser verifica ordine delle batterie, colonne separate, numeri finiti non negativi e campioni completi. Accetta BOM UTF-8, CRLF, righe vuote dopo l'intestazione e virgola decimale. Gli errori riportano la riga fisica del CSV.

CodeBlocks compila lo stesso CPP di Visual Studio e usa 72V-Battery-AI-Model-Trainer come directory di esecuzione, quindi gli stessi dati e file modello nella sottocartella 72V-Battery-S11.
La costante training_samples resta 323 e deve corrispondere ai campioni del nuovo dataset: il programma controlla il conteggio prima di procedere.

Test da 72V-Battery-AI-Model-Trainer con MinGW:

    g++ -std=c++11 -static tests/csv_parser_test.cpp -o csv_parser_test.exe -luser32
    ./csv_parser_test.exe

I test verificano il formato nuovo, il rifiuto di quello vecchio, i dati incompleti/non validi e l'associazione dei valori a training e inferenza. Non generano modelli.
