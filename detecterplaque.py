import cv2
import imutils
import numpy as np

lic_data = cv2.CascadeClassifier('data/alexnet/haarcascade_russian_plate_number.xml')

def plaquecasc(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    number = lic_data.detectMultiScale(img,1.2)
    return number

def norm(t):
    return -t[2]*t[3]

def nearcenter(c):
    rect = cv2.minAreaRect(c)
    x,y = rect[0]
    return (x-250)**2+(y-250)**2

def plaque(img):
    factorbig = 1
    if img.shape[1] < 500:
        factorbig = img.shape[1]/500
        img = imutils.resize(img, width=500)
    ##imshow(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    number = lic_data.detectMultiScale(gray,1.2)
    number = sorted(number,key=norm,reverse=True)[:1]
    x=0
    y=0
    marge = 0
    for numbers in number:
        (x,y,w,h) = numbers
    
    if len(number) == 0:
        image = img.copy()
    else:
        marge = 10
        heigth, width, channels = img.shape
        image = img[max(0,y-marge):min(y+h+marge,heigth),max(0,x-marge):min(width,x+w+marge)].copy()
    
    factor = 1
    if image.shape[1] > 500:
        factor = image.shape[1]/500
        image = imutils.resize(image, width=500)
    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #imshow(gray)
    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 10))
    erosion = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    #imshow(erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    erosion2 = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    #imshow(erosion2)
    contrast = 255/(max(erosion2.flatten())-min(erosion2.flatten()))
    contrasted = ((erosion2-min(erosion2.flatten()))*contrast).astype(np.uint8)
    #imshow(contrasted)
    final = cv2.threshold(contrasted, 190, 255, cv2.THRESH_BINARY)[1]
    #imshow(final)
    # Find Edges of the grayscale image
    edged = cv2.Canny(final, 170, 200)
    #imshow(edged)
    # Find contours based on Edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    cnts = sorted(cnts, key = nearcenter)
    NumberPlateCnt = None #we currently have no Number plate contour
    
    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            rect = cv2.minAreaRect(c)
            l,h = rect[1]
            if l < h:
                l,h = h,l
            if h!= 0 and l/h >2 and l/h<10 and len(approx)<=5:  # Select the contour with 4 corners
                
                NumberPlateCnt = c #This is our approx Number Plate Contour
                x2,y2,w2,h2 = cv2.boundingRect(c)
                ROI = img[y2:y2+h2, x2:x2+w2]
                break
    if NumberPlateCnt is None:
        return None
    else: return ((NumberPlateCnt*factor + np.array([x-marge,y-marge]))*factorbig).astype(int) 

def imshow(image):
    while True:
        cv2.imshow("img",image)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cv2.destroyAllWindows()


img = cv2.imread("data/alexnet/car10.jpg")
cv2.drawContours(img, [plaque(img)], -1, (0,255,0), 2)

#imshow(img)
