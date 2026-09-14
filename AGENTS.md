# Regole Progetto

## Contesto Del Repository

- Questo repository `72Volts-Battery-Health-Check` e' un contesto di test, generazione modello, inferenza e verifica sperimentale.
- Non considerare questo repository come il firmware di produzione del sistema.
- `72V-Battery-AI-Model-Trainer` (72 Volts Battery AI Model Trainer, dove AI significa Artificial Intelligence) e' il progetto usato per importare il CSV, addestrare il modello e verificarne le predizioni.
- `72V-Battery-AI-Model-Tester` e' il progetto di test del modello sul microcontrollore, utile anche per sperimentare strategie come la cosine strategy sui vettori.
- `72V-Battery-AI-Model-Tester` non e' il software che va in produzione sul sistema.
- Il vero software di produzione da verificare per compilazione, memoria e anomalie e' `Cell Battery Health Check`.
- Se una richiesta riguarda il firmware di produzione, prima di compilare, analizzare o modificare codice verificare esplicitamente di essere nel progetto `Cell Battery Health Check`.
- In questo repository non fare assunzioni di produzione su memoria, librerie o target hardware salvo richiesta esplicita.
