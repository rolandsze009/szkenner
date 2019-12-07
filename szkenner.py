import numpy as np 
import cv2


#Bekérjük a képet
img = cv2.imread("test.jpg",1)
#átméretezzük
img = cv2.resize(img,(800,1200))
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
image,contours,hierarchy = cv2.findContours(eldetektaltkep,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
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
    if len(kozelito) == 4
    vegsokonturok = kozelito
    break


cv2.waitKey(0)
cv2.destroyAllWindows()