# Conversational Bot as Parliamentary Assistant
## Overview
Questo progetto ha lo scopo di realizzare un chatbot  come strumento innovativo per la predisposizione di proposte di legge e di atti di indirizzo e di controllo da parte di deputate e deputati.
L’obiettivo di tale studio di ricerca è quello di fornire un mezzo basato su IA generativa che sia in grado di facilitare il processo decisionale legislativo, grazie alla possibilità di mettere a disposizione dei legislatori una vasta gamma di fonti eterogenee e permettendo di contestualizzare le loro proposte di legge nel panorama normativo, economico e sociale.
L'output atteso da questo prototipo di IA è quello di fornire un supporto prezioso all'attività parlamentare in materia di parità di genere, contribuendo così a promuovere politiche pubbliche più inclusive e equilibrate. 
Nel lungo termine ci si pone l’obiettivo di estendere le funzionalità del sistema di IA generativa ad ogni altro ambito di interesse politico.
Inoltre, la proposta tiene conto del pluralismo linguistico, incluse le minoranze linguistiche italiane, nonché le principali lingue veicolari: l'inglese, il francese e il tedesco. 
Il Dataset fornito è stato suddiviso in quattro sezioni, ognuna delle quali contenente documenti riguardanti norme giuridiche, atti parlamentari e diverse sentenze. Sulla base di questo dataset è possibile addestrare il chatbot di modo che fornisca delle risposte consistenti e coerenti con la natura del problema.

## Installation 
1. Per generare il vector_store è necessario avviare lo script da terminale inserendo il seguente comando, sostituendo NOME_CARTELLA con il nome della cartella che presenta i documenti per la base di conoscenza:
`python vector_store_creation.py .\{NOME_CARTELLA}`

2. Installare `ollama` aprendo un terminale e digitando il seguente comando: 

`pip install ollama`

3. Installare il modello `llama3` inserendo il seguente comando nel terminale:

`ollama pull llama3`

4. Per avviare il chat_bot bisogna inserire nel terminale il seguente comando: 

`streamlit run chatbot_streamlit_combined.py`

## Caratteristiche
### 1. Dataset:
Per la realizzazione del prototipo, abbiamo concentrato la nostra selezione sulle tematiche dell'empowerment femminile, dei diritti sociali e della regolazione attuativa del PNRR all'interno dell'area dell'uguaglianza di genere.
Il dataset è stato suddiviso in sottocartelle ognuna contente file pdf su differenti argomenti. Di seguito la struttura:
1.	Normazione:
    1.	Normazione dal 1948 al 2024:
        1.	Atti pari opportunità - empowerament femminile e pnrr: 24 documenti .pdf ed 1 .docx
        2.	Leggi parità di genere - diritti sociali: 23 documenti .pdf
    1.	Proposte di legge XVIII e XIX Leg: 
        1.	Proposte di legge Leg XVII: 19 documenti .pdf ed 1 .docx 
        2.	Proposte di legge Leg XIX: 5 documenti .pdf
    3.	Proposte di iniziativa legislativa popolare dal 2001 al 2024: 4 documenti .pdf
La selezione delle proposte di legge di iniziativa popolare (indicata con la lettera "c") è stata effettuata considerando l'identità nella redazione di un articolato normativo, indipendentemente dal soggetto proponente, che potrebbero essere cittadini, parlamentari o altri soggetti legittimati ai sensi dell'articolo 71 della Costituzione. È importante che lo strumento risponda alle esigenze specifiche dei deputati e delle deputate.
Nel  nome file dei rispettivi documenti, si è stato riportato se la proposta è stata approvata o no, dato che potrebbe essere utile per verificare la verosimiglianza dell’approvazione di una proposta di legge.
3.	Giurisprudenza:
    a.	Sent. Corte costituzionale: 18 documenti .pdf
    b.	Sent. Corte di giustizia UE: 9 documenti .pdf
    c.	1 documento .docx sommario della cartella Giurisprudenza
4.	Dossier: 31 documenti .pdf 
5.	Atti indirizzo e controllo Parlamento:
    a.	Atti indirizzo e controllo XVIII Legislatura: 8 documenti .pdf
    b.	Atti indirizzo e controllo XIX legislatura: 12 documenti .pdf
Questo approccio multidimensionale consente di accedere a una vasta gamma di informazioni legislative e giuridiche rilevanti per la promozione dell'uguaglianza di genere e per la predisposizione di proposte legislative mirate e informate.

### 2. ChatBot 
Il modello si basa su chiamate all'API di llama3 attraverso una maipolazione del prompt, in prima istanza: 
1. Attraverso la funzione `PromptTemplate()` abbiamo modificato il prompt in modo tale da permettere la generazione di cinque diverse interpretazioni della domanda dell'utente da presentare al modello. Questo approccio permette di superare eventuali errori di battitura o domande poco chiare.
2. Definizione del `MultiQueryRetriever.from_llm()`, che permette di usare le 5 domande per recuperare un set di documenti rilevanti per ogni documento e prende un unione di tutte le queries per avere un set più grande di documenti rilevanti, al fine di superare le limitazioni delle metriche di similarità.
3. Definizione della memoria, attraverso `ConversationBufferWindowMemory()`, che permette di creare una lista delle interazioni. Esso usa le ultime k=4 interazioni, e permette di tenerne traccia senza rendere la finestra di memoria troppo grande. 
4. Creazione del chat_bot attraverso `ConversationalRetrievalChain.from_llm()`.
5. Definzione del prompt finale per ottnere la risposta, tale prompt include sia l'ultima domanda dell'utente sia la storia recente conservata nella memoria. 

