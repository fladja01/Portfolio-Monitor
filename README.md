# Fladjellux Fund Portfolio Monitor

Dashboard HTML aggiornata automaticamente da GitHub Actions con prezzi Yahoo Finance via `yfinance`.

## Cosa fa

Ogni aggiornamento:

1. legge `data/positions.csv`;
2. legge `data/instruments_master.csv`;
3. scarica i prezzi più recenti da Yahoo Finance;
4. converte i controvalori in EUR;
5. ricalcola pesi correnti;
6. calcola rendimento da ultimo aggiornamento;
7. calcola rendimento da acquisto;
8. calcola contribution come `peso × rendimento`;
9. genera `docs/index.html`;
10. salva uno snapshot in `data/last_snapshot.csv`.

## Come usarlo su GitHub

1. Crea una nuova repository GitHub.
2. Carica tutti i file di questa cartella.
3. Vai su **Settings → Pages**.
4. In **Build and deployment**, scegli:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/docs`
5. Vai su **Actions**.
6. Esegui manualmente il workflow **Update portfolio dashboard** la prima volta.
7. Dopo qualche minuto, apri il link GitHub Pages.

## Aggiornamento automatico

Il file `.github/workflows/update-dashboard.yml` aggiorna la dashboard tre volte al giorno nei giorni feriali.

Puoi anche fare update manuale:

**Actions → Update portfolio dashboard → Run workflow**

## Dove modificare i portafogli

Modifica:

- `data/positions.csv`
- `data/instruments_master.csv`
- `data/portfolio_settings.json`

Ogni strumento deve essere classificato come:

- `PIC`
- `PAC`

Non esiste una terza categoria.

## Formula della contribution

La dashboard usa:

```text
contribution = peso × rendimento
```

Esempio:

```text
peso = 7.16%
rendimento = +2.10%

contribution = 7.16% × 2.10% = +0.150%
```

## Nota dati

`yfinance` è comodo per demo/prototipi, ma non è una fonte istituzionale. Per uso professionale, sostituire il data provider con Bloomberg, Refinitiv, FactSet, EODHD o altro provider autorizzato.
