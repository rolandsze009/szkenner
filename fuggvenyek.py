import numpy as np 
import cv2

kep = 0

def kepinput(kepneve_kiterjesztessel):
 
    #Bekérjük a képet
    img = cv2.imread(kepneve_kiterjesztessel,1)
    #átméretezzük
    img = kep_atmeretezes_aranyosan(img, magassag = 700)
    cv2.imshow("image.jpg",img)
    #szürkeárnyalati konverió Canny detektorhoz
    szurkekep = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("szurkekep.jpg",szurkekep)
    #későbbi éldetektálást segítő Gauss szűrő alkalmazása Canny detektorhoz
    #5X5ös kernellel
    simitottkep = cv2.GaussianBlur(szurkekep, (5,5),0)
    cv2.imshow("simitottkep.jpg",simitottkep)
    # Canny detektor 35-60 as threshold-dal.
    eldetektaltkep = cv2.Canny(simitottkep,35,60)
    cv2.imshow("eldetektaltkep.jpg",eldetektaltkep)

    #A kontúrokat egyenlő hierarchia szinten kezeljük (cv2.RETR_LIST),
    #valamint a kontúrvonalak végpontjait vesszük- (cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(eldetektaltkep,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #rendezzük a megtalált kontúrokat(contours) területi nagyságuk szerint (key=cv2.contourArea)
    # a legnagyobbal kezdve (reverse = True), mert alapból növekvő sorrendben rendez
    contours = sorted(contours, key=cv2.contourArea, reverse = True)

    #Meghívjuk a ciklust az összes kontúrra
    for x in contours:
        #visszaadja egy zárt (True) alaknak az ivhosszat (cv2.arcLength)
        #0.01re állítva epsilon értékét: ami a maximum távolságot jelenti
        #a kontúrtól, jelen esetben ez 1% 
        ivhossz = 0.01*cv2.arcLength(x,True)
        #közelítő értéket ad vissza az ivhossz es a kontúrokból úgy, hogy
        #egy zárt (True) közelítő kontúrvonalat keresünk
        kozelito = cv2.approxPolyDP(x,ivhossz,True)
        #visszaadja az elemek számát egy tárolóból
        if len(kozelito) == 4:
            vegsokonturok = kozelito
            break

    
    kozelito=konturrendezo(vegsokonturok) #részletesen leírva a függvény definíciójánál

    #részletesen leírva a függvény definíciójánál 
    aktualiskepszama()
    print(kozelito)
    
    #Itt állítjuk be melyik méretarányban szeretnénk szkennelni
    ################################ Méret Állítás #############################
    
    #pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  # egyenlő oldalú lap
    #pts=np.float32([[0,0],[496,0],[496,702],[0,702]])  # A4-es méret álló
    #pts=np.float32([[0,0],[702,0],[702,496],[0,496]])  # A4-es méret fekvő  

    op=cv2.getPerspectiveTransform(kozelito,pts)  #vesszük a kép felülnézetét
    
    #dst=cv2.warpPerspective(img,op,(800,800)) # egyenlő oldalú lap
    #dst=cv2.warpPerspective(img,op,(496,702)) # A4-es méret álló
    #dst=cv2.warpPerspective(img,op,(702,496)) # A4-es méret felvő 

    ################################ Méret Állítás #############################

    konturpontok = cv2.drawContours(img, vegsokonturok, -1, (102,255,178), 20)
    cv2.imshow("Konturpontok",konturpontok)
    cv2.imshow("Szkennelt",dst)

    elesitett(dst)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    # konvolúciót használunk egy élesítéshez használt kernel beállítással, a kép élesítéséhez
def elesitett(dst):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    im = cv2.filter2D(dst, -1, kernel)
    cv2.imshow("Elesitett", im)
    
    # bal-felső -> jobb-felső -> jobb-alsó -> bal-alsó sorrendben (x,y) koordinátákat felvesszük ebben
    # a sorrendben rendezzük.
def konturrendezo(anegypont):
    # felvesszük a végső kontúrpontokat új "négyzet" 2d-s "4 soros, 2 adattagos" alakba
    anegypont = anegypont.reshape((4,2))
    #felveszünk egy új 2d-s tömböt float32 adattípussal és feltöljük nullákkal
    rendezettnegypont = np.zeros((4,2),dtype = np.float32)
    
    add = anegypont.sum(1)
    # bal-felső pont meglesz az adott végsőkontúrok tömbbeli legkisebb x+y koordináták szummájával
    rendezettnegypont[0] = anegypont[np.argmin(add)]
    # jobb-alsó pont meglesz az adott végsőkontúrok tömbbeli legnagyobb x+y koordináták szummájával
    rendezettnegypont[2] = anegypont[np.argmax(add)]

    diff = np.diff(anegypont,axis = 1)
    # jobb-felső pont meglesz az adott végsőkontúrok tömbbeli legkisebb x-y koordináták különbségével
    rendezettnegypont[1] = anegypont[np.argmin(diff)]
    # jobb-felső pont meglesz az adott végsőkontúrok tömbbeli legnagyobb x-y koordináták különbségével
    rendezettnegypont[3] = anegypont[np.argmax(diff)]
    # visszaadjuk a megadott sorrendben rendezett kontúrpontok koordinátáit
    return rendezettnegypont

    # kiírja az aktuálisan feldolgozás alatt álló kép sorszámát, ha több kép lett megadva a mainben
    # akkor mindegyik képhez kiírja a "sorszámát" utána pedig hogy hol található a kontúrpontok koordinátái
    # ez a funkció segít összevetni a manuálisan meghatározott kontúrpontokhoz képest a program pontosságát
    # bal-felső -> jobb-felső -> jobb-alsó -> bal-alsó sorrendben (x,y) formátumban kiírja a konzolra
def aktualiskepszama():
    global kep
    kep += 1
    print(str(kep) + ". kep konturpontjai:")
    

    #függvény a kép átméretezéséhez arányosan
def kep_atmeretezes_aranyosan(kep, szelesseg = None, magassag = None, inter = cv2.INTER_AREA):
    # felvesszük a meret nevű változót amibe az átméretezett kép méretét tesszük bele
    meret = None
    # vesszük a kép magasságát és szélességét
    (m, sz) = kep.shape[:2]

    # ha a paramétereknél nem adunk meg sem magasságot sem szélességet akkor
    #az eredeti kép méretét kapjuk
    if szelesseg is None and magassag is None:
        return kep

    # ha csak a magasság van megadva akkor kiszámolja a szélességet hozzá arányosan
    # és eltárolja meret nevű változóban
    if szelesseg is None:

        r = magassag / float(m)
        meret = (int(sz * r), magassag)

    # egyébként a szélesség van megadva és kiszámolja a magasságot hozzá arányosan
    else:
        r = szelesseg / float(sz)
        meret = (szelesseg, int(m * r))

    # újraméretezi a képet
    ujrameretezettkep = cv2.resize(kep, meret, interpolation = inter)

    # visszaadja az újraméretezett képet
    return ujrameretezettkep