# Levende bilder

Malerier og fotografier med ekte parallakse. En AI-modell (Depth Anything 3) lager et
dybdekart av bildet, og en WebGL2-shader bruker kartet til å forskyve pikslene når du
flytter musa eller vipper telefonen. Forgrunnen beveger seg mer enn horisonten.

Alt kjører i nettleseren uten backend. Demoen har 11 verk:

- **Malerier og fotografier fra Nasjonalmuseet** (fotografier av Knud Knudsen og Anders Beer
  Wilse). De hentes fra IIIF-tjenesten på samme måte som `ImageUtilHelperService.GetIiifImageUrl`
  i NamWeb.
- **Et av Munchs egne fotografier fra MUNCH**, hentet fra munch.no. Nasjonalmuseet har ikke
  Munchs fotografier.

Malerier med tydelig perspektiv og klare lag (interiører, figurgrupper, landskap med forgrunn)
gir best effekt. Ekspresjonistiske eller nattlige malerier med mye himmel fungerer dårlig,
fordi modellen tolker malt himmel som nær. `tools/artworks.json` lister hvilke verk som er
testet og tatt ut, og hvorfor.

## Kom i gang

```powershell
npm install
npm run dev            # http://localhost:5174
```

Verkene og dybdekartene ligger allerede i `web/art/`. Du trenger bare Python-delen hvis du
vil legge til verk eller lage kartene på nytt.

### Hent fra samlingen (på demosiden)

```powershell
npm run api            # tools/server.py på :5175, Vite videresender /api dit
```

Når serveren kjører, får demosiden knappen **«Hent fra samlingen»**. Den fungerer slik:

1. Du søker i Nasjonalmuseets samling (samme søke-API som nasjonalmuseet.no) og filtrerer på
   maleri, fotografi, tegning eller grafikk.
2. Du klikker på et verk.
3. Serveren henter bildet via IIIF, lager dybdekartet med Depth Anything 3 på GPU-en og
   legger verket til i `web/art/works.json` og `tools/artworks.json`.

Første bilde tar litt lenger tid, fordi modellen lastes inn. Etter det tar hvert bilde
cirka 5–10 sekunder. Også verk som er vernet av opphavsrett, kan hentes. Rettighetsmerket
fra verkssiden (for eksempel «© Fjell, Kai/BONO») vises da i bildeteksten i stedet for
CC BY. Slike verk er bare til intern demo; offentlig bruk må klareres med rettighetshaver.
Uten serveren skjules knappen, og siden fungerer som før. `python tools/server.py`
alene leverer også hele siden på http://localhost:5175.

### På nett (GitHub Pages)

Hver push til `main` publiserer `web/` til https://fredrikskaug.github.io/levende-bilder/
(`.github/workflows/pages.yml`). Det er bare statiske filer, så siden virker uten PC-en, også
på mobil (https gir tilgang til gyro og kamera). «Hent fra samlingen» krever Python-serveren og
vises ikke der; nye verk lages lokalt og pushes. Siden er offentlig, men merket `noindex`, så
søkemotorer ikke skal vise den. Fonter, logo og vernede verk er bare til demo.

### Mobil (gyro og hodesporing)

Gyroskop og kamera krever HTTPS på telefonen. Enklest er en ngrok-tunnel, som også virker på
kontornett der telefonen ikke når PC-en direkte:

```powershell
npm run dev            # i ett vindu
npm run tunnel         # i et annet: gir en https://….ngrok-free.app-adresse
```

Åpne adressen på telefonen og trykk «Visit Site» på ngrok sin advarselsside (én gang).
`--host-header=rewrite` gjør at Vite tror forespørslene kommer fra localhost. Adressen er
offentlig så lenge tunnelen går, så ikke del den, og stopp tunnelen etterpå.

På samme nett uten tunnel: `npm run dev:https` og `https://<din-ip>:5174`, godta
sertifikatadvarselen.

- **Bevegelse**: Trykk på bevegelsesknappen øverst (iOS spør om tillatelse). Uten gyro kan du
  dra med fingeren.
- **Følg hodet**: Innstillinger → «Følg hodet». Hold telefonen 30–40 cm fra ansiktet under
  kalibreringen. I Safari kjører ansiktssporingen i en Web Worker. I Chrome, Arc og innebygde
  nettlesere på iPhone feiler MediaPipe der («Can't find variable: document»), og da kjører den
  på hovedtråden i stedet.

## Legge til verk og lage dybdekart

```powershell
.\tools\setup.ps1                      # Python 3.12-venv med CUDA-torch og Depth Anything 3 (-Cpu for bare CPU)
.\.venv\Scripts\python tools\fetch_artworks.py   # henter bilder (1500 px) og metadata via IIIF
.\.venv\Scripts\python tools\make_depth.py       # lager web/art/<verk>/depth.png
```

1. Legg til objekt-ID-en (for eksempel `NG.M.00844` eller `NMK.2007.6426`) i
   `tools/artworks.json`. Velg verk der opphavspersonen døde for mer enn 70 år siden. Objekt-ID-er
   kan du finne med samlingssøket som nettsiden bruker:
   `https://www.nasjonalmuseet.no/api/collection/ulc/searchquery/no?query=Wilse&object-name=Fotografi`.
2. `fetch_artworks.py` leser verkssiden, finner `originalFile` og bygger IIIF-URL-en. Hvis
   skriptet velger feil bilde (for eksempel et detaljfoto), sett `"file"` i `artworks.json`.
   Fotografier har ofte passepartout eller tekst. Fjern det med `"crop": [venstre, topp, høyre, bunn]`.
   For MUNCH-bilder settes `"source": "munchmuseet"`, og metadataene skrives inn i
   `artworks.json`.
3. `make_depth.py` kjører `DA3MONO-LARGE`, tar snittet av originalen og et speilvendt bilde,
   presser himmelen bakover, vokser forgrunnen litt og lagrer en 8-bits PNG.
   Det tar rundt 2 sekunder per maleri på en RTX 4050.

`focus` og `strength` per verk i `artworks.json` gir startverdiene for fokusplan og styrke.

## Slik virker det

- **Dybde**: `DA3MONO-LARGE` er den monokulære modellen i Depth Anything 3-familien (Apache 2.0).
  Hovedmodellene (`DA3-LARGE` o.l.) er laget for geometri fra flere bilder og ser et maleri som
  en flat flate, så de egner seg ikke.
- **Himmel**: DA3 tolker ofte malt himmel som nær. Vi bruker modellens sky-sannsynlighet som en
  myk vekt og sprer den oppover (det som er over himmel, er også himmel).
- **Shader** (`web/js/shaders.js`): «Enkel» er `uv += (dybde − fokus) · tilt · styrke`.
  «Okklusjon» følger synslinjen gjennom dybdekartet fra forgrunnen og bakover
  (parallax occlusion mapping), slik at forgrunnen dekker bakgrunnen riktig.
  Antall steg tilpasses hvor langt bildet faktisk flyttes.
- **Pust**: Når ingen rører musa, går kameraet i en langsom bane.
- **Støv i lyset**: Partikler på ulike dybder følger samme parallakse, skjules bak nærmere
  flater og vises bare i lyse partier.
- **Dolly zoom**: Forgrunnen vokser mens bakgrunnen krymper, med fokusplanet låst.
- **Følg hodet** (`C`, eller `?head` i URL-en): webkameraet følger hodet ditt med MediaPipe
  Face Landmarker, og perspektivet følger hodet. Hodet måles i centimeter: pupillavstanden din
  (cirka 6,3 cm) er linjalen i kamerabildet. Står du der du satt under kalibreringen, ser du
  maleriet rett forfra.

  Motivet presses inn i et grunt relieff rundt skjermen. Fokusplanet står stille på skjermen;
  det som er nærmere, kommer ut mot deg og glir mot hodebevegelsen din, og det som er lenger
  unna glir med den. Dybden kommer av at lagene beveger seg i forhold til hverandre. «Dybde»
  (standard 10 %) er relieffets dybde som andel av bildets bredde, så det ser likt ut på mobil
  og PC: en fast dybde i centimeter ble tre ganger så dyp på mobil som på PC i forhold til
  bildet. Bildet beskjæres bare noen få prosent, og høyst 12 % per side; beveger du deg lenger,
  toner kantene ut. «Vis kamerabildet» viser deg selv i hjørnet.

  Hvorfor ikke et ekte vindu: Vi prøvde med ekte avstander, der fjerne fjell glir like langt som
  hodet ditt. På en flat skjerm ser begge øynene det samme bildet, og de forteller hjernen at alt
  er på skjermen. Det som glir over skjermen, oppleves da som at bildet flytter seg. I tillegg
  måtte bildet beskjæres kraftig, så på en bærbar skjerm så man nesten ingenting gjennom vinduet.

  Slik er det gjort jevnt (etter [desktop-vr](https://github.com/jasondecamp/desktop-vr) og
  [trompeloeil](https://github.com/vivien000/trompeloeil)):
  - Ankerpunktet ligger mellom øynene, og de indre øyekrokene veier mer enn irisene, fordi de
    står stille når du blunker eller ser rundt.
  - Face Landmarker *sporer* ansiktet fra bilde til bilde i stedet for å finne det på nytt, og
    det gir mindre hopping.
  - Et One Euro-filter holder bildet rolig når du står stille og følger raskt når du beveger deg.
  - Sporingen kjører i en Web Worker (`headtrack.worker.js`), slik at den aldri stopper tegningen.
    Selve sporingen ligger i `facetracker.js`, slik at den også kan kjøre på hovedtråden der
    MediaPipe ikke virker i en worker.
  - Shaderen bruker et fast antall steg per maleri, slik at kantene ikke hopper når kameraet beveger seg.

  Analysen skjer lokalt i nettleseren, og ingenting lagres eller sendes.

  **Kalibrering** (åpnes første gang, eller Innstillinger → Hodesporing → «Kalibrer»), cirka 15 s:
  1. Sitt i ro og se på prikken. Det setter hvor «rett forfra» er og hvor stor pupillavstanden
     din er i kamerabildet, og måler hvor mye sporingen skjelver. «Ro» settes akkurat lavt nok
     til å skjule skjelvingen.
  2. Beveg hodet fra side til side. Det måler hvor langt (sidelengs og opp/ned hver for seg) og
     hvor fort du beveger deg. Beskjæringen tilpasses bevegelsesområdet ditt, og «respons»
     settes etter farten din.
  3. Skjermen blinker. Det måler forsinkelsen i skjerm og kamera, og prediksjonen settes til
     80 % av den.

  Avstanden din regnes ut fra pupillavstanden og en antatt kameravinkel: 60° for webkamera og
  74° for frontkameraet på iPhone (langs bildets lange side).

  **Måle og justere forsinkelsen** (Innstillinger → Hodesporing, synlig når kameraet går):
  - Grafen viser hva kameraet så (grått) mot det som tegnes (gult). Avstanden mellom kurvene er
    etterslepet, og det måles automatisk når du beveger hodet sidelengs.
  - «Mål forsinkelse» blinker skjermen svart/hvitt og måler når kameraet ser ansiktet ditt bli
    lyst. Det er forsinkelsen i skjerm og kamera, som ellers ikke kan måles innenfra. Det virker
    best i et dempet rom.
  - Glidebryterne styrer filteret og prediksjonen, og verdiene huskes i nettleseren:
    - «Ro i stillstand»: One Euro `minCutoff`
    - «Respons i bevegelse»: `beta`
    - «Prediksjon»: hvor mange millisekunder bevegelsen skrives frem
  - «Ansiktsmodellen kjører på»: Standard er å måle både GPU og CPU på de første kamerabildene
    med ansikt og beholde den raskeste. På en bærbar PC med to skjermkort bruker nettleseren
    som regel det integrerte, som deles med tegningen av maleriet. Da tar ansiktsmodellen
    40–60 ms per bilde, og det er den største forsinkelsen. Sett nettleseren til
    «Høy ytelse» under Windows-innstillinger → System → Skjerm → Grafikk, slik at både sporingen
    og tegningen går på det kraftige skjermkortet.
  - Best avstand er 50–70 cm med godt lys forfra. I mørke senker kameraet bildefrekvensen, og da
    øker forsinkelsen. Når ingen står
  foran, tar mus eller «pust» over igjen, så `?kiosk&head` fungerer godt på en skjerm i museet.
  Modellen (`web/assets/models/face_landmarker.task`) finner ansikter på opptil cirka 2 meters
  avstand. Biblioteket hentes fra jsDelivr, så kiosken trenger nett.
- **Lag klipp**: Tar opp en sømløs loop på 6 sekunder (4:5, 9:16, 1:1 eller 16:9) med bildetekst,
  klar for sosiale medier.

## Visningsmoduser

| URL | Bruk |
|---|---|
| `/?work=ng-m-00939` | Åpne et bestemt verk |
| `/?embed` | Toppbilde på en verksside: så stort som mulig i boksen (aldri beskåret), uten grensesnitt, med pust |
| `/?kiosk` | Skjerm i museet: fyller skjermen uten grensesnitt, med pust, bytter verk hvert 25. sekund (piltastene virker også) |
| `/?kiosk&head` | Som over, men perspektivet følger hodet til den som står foran (webkamera) |

Hurtigtaster: `←` `→` bytt verk · `O` original · `M` teknikk · `D` dybdekart · `B` pust · `S` støv ·
`Z` dolly zoom · `F` fyll skjermen (så stort som mulig uten beskjæring; grensesnittet skjules når musa er i ro) · `C` følg hodet · `R` lag klipp · `H` skjul grensesnitt.
Dobbeltklikk på maleriet for å sette fokusplanet der.

## Struktur

```
tools/
  artworks.json        kuraterte verk med startverdier
  fetch_artworks.py    IIIF-nedlasting og metadata
  make_depth.py        Depth Anything 3 → depth.png
  server.py            søk i samlingen + dybdekart på forespørsel (demosiden)
  setup.ps1            Python-miljø
web/
  assets/              MuseetSans, museet-icons og logo fra NamWeb
  index.html, css/, js/
    shaders.js         GLSL: parallakse, dybdevisning, støv
    renderer.js        WebGL2-oppsett og tegning
    input.js           mus, berøring og gyro
    headtrack.js       hodesporing med webkamera: hodeposisjon i cm og One Euro-filter
    facetracker.js     ansiktssporing (MediaPipe Face Landmarker), i worker eller på hovedtråden
    calibration.js     kalibrering: rett forfra, filter, forsinkelse
    headtrack.worker.js  kjører facetracker.js utenfor hovedtråden
    main.js            tilstand, grensesnitt, opptak
  art/works.json       generert manifest
  art/<verk>/          image.jpg + depth.png
```

## Design

Grensesnittet følger samlingssidene på nasjonalmuseet.no. Fontene (MuseetSans), ikonfonten
(`museet-icons`) og logoen er kopiert fra `NamWeb/Frontend/Resources` til `web/assets/`,
og fargene er hentet fra `Styles/custom/base/colors`. Fontene er lisensiert til Nasjonalmuseet,
så demoen er bare til internt bruk og skal ikke publiseres offentlig uten avklaring.

Fra museumsetikken (ICOM, og Nasjonalmuseets egen KI-praksis) har vi tatt med:

- et synlig merke på maleriet: «Dybdeeffekt laget med KI»
- «Vis original» for å se verket uten effekt
- redusert bevegelse når `prefers-reduced-motion` er slått på
- merking også i videoklippene

## Bilderettigheter

Verkene er falt i det fri. Reproduksjonene fra Nasjonalmuseet er CC BY 4.0, og fotografen
oppgis i bildeteksten. Reproduksjonene av Munchs fotografier er «© Munchmuseet», så de
kan bare brukes internt uten avtale med MUNCH.
