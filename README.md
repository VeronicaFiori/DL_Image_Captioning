# Un Sistema di Image Captioning Multimodale su Flickr8k

> Generazione automatica di descrizioni testuali a partire da immagini utilizzando
> modelli multimodali di grandi dimensioni e valutazione combinata tramite metriche
> automatiche e giudizio Vision–Language.


## Obiettivo del progetto

Il progetto affronta il problema dell’**Image Captioning**, ovvero la generazione
automatica di descrizioni testuali coerenti e semanticamente fedeli a partire da immagini.

In particolare, gli obiettivi principali del progetto sono:
- analizzare il comportamento di un **modello multimodale di stato dell’arte (BLIP-2)** applicato al dataset Flickr8k;
- studiare la qualità delle caption generate;
- ridurre il rischio di **allucinazioni semantiche** tramite strategie di generazione controllata;
- valutare le prestazioni del sistema sia tramite **metriche automatiche standard**
  sia tramite una **valutazione multimodale basata su un modello Vision–Language**.


## Background e tecniche utilizzate

### Modelli utilizzati
- **BLIP-2 + Flan-T5-XL**: backbone multimodale per Image Captioning
- **Qwen2.5-1.5B-Instruct**: riscrittura controllata dello stile linguistico
- **Qwen2-VL-7B-Instruct**: giudice multimodale per la valutazione della fedeltà
  immagine–testo

### Strumenti
- Python
- PyTorch
- HuggingFace Transformers
- Metriche automatiche per Image Captioning (BLEU, CIDEr, ROUGE)
- Le dipendenze sono presenti in requirements.txt

### Dataset
- **Flickr8k**
  - 8.000 immagini naturali
  - 5 caption per immagine
  - Suddivisione standard in training, validation e test set

## Esperimenti e valutazione

Il progetto prevede una valutazione quantitativa basata su metriche automatiche, Bleu(1/2/3/4), CIDEr e ROUGE-L, e una valutazione qualitativa basata su un giudice multimodale, Qwen2-VL-7B-Instruct, il quale richiede una risposta strutturata in formato JSON che includa un giudizio binario di fedeltà, un punteggio numerico su scala discreta, l’eventuale elenco di allucinazioni individuate e una breve motivazione testuale.




## Analisi dei risultati

I risultati ottenuti mostrano che il sistema è in grado di generare caption coerenti e linguisticamente corrette, con prestazioni in linea con quanto atteso per modelli pre-addestrati valutati su Flickr8k.

Le metriche automatiche evidenziano una buona capacità di catturare il contenuto
lessicale e semantico delle immagini, mentre la valutazione multimodale conferma che una parte significativa delle caption è semanticamente fedele al contenuto visivo.
L’analisi con Qwen2-VL-7B-Instruct mette inoltre in evidenza i limiti delle metriche
puramente lessicali, giustificando l’uso di una valutazione multimodale affiancata.
