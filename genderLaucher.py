from copy import deepcopy
import sys, pygame
import pygame.camera
from pygame.locals import *
import constanst as MAPCOLORING
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from keras_preprocessing.image import img_to_array
from keras.models import load_model
from gender_detect import detect_image

def main():
    global FPSCLOCK, DISPLAYSURF, ILLU, MAP, BASICFONT, BUTTONS, MAXSTATE, COLOURS, RUNTIMEFONT, CAM

    COLOURS = ['red', 'green', 'blue', 'yellow']
    
    pygame.init()
    pygame.camera.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((MAPCOLORING.WINWIDTH, MAPCOLORING.WINHEIGHT))
    BUTTONS = [('Upload Image'), ('Save Image'), ('Realtime Detection')]
    MAXSTATE = len(BUTTONS) - 1
    pygame.display.set_caption('Face and Gender Detection')
    BASICFONT = pygame.font.Font('assets/fonts/8-BITWONDER.TTF', 18)
    RUNTIMEFONT = pygame.font.Font('assets/fonts/FreeSansBold.ttf', 18)
    root = tk.Tk()
    root.withdraw()

    camlist = pygame.camera.list_cameras()
    if camlist:
        CAM = pygame.camera.Camera(camlist[0],(640,480))


    run()

def run():
    menuNeedsRedraw = True
    state = 0
    image = cv2.imread('assets/images/sample.jpg')
    map = pygame.image.load('assets/images/sample.jpg')
    mapNeedsRedraw = True
    uploadedPhoto = True
    runtime = 0
    videoCapture = False
    output = None

    model = load_model('gender_detect.model')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')

    while True: 
        cursorMoveTo = 0
        if uploadedPhoto == False:
            for event in pygame.event.get():
                if event.type == QUIT:
                    terminate()
                elif event.type == KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if state == 0:
                            temp = uploadPhoto()
                            if temp is not None:
                                
                                image = temp
                                uploadedPhoto = True
                                if videoCapture == True:
                                    videoCapture = False
                                    CAM.stop()
                        elif state == 1:
                            if output is not None:
                                success = savePhoto(output)
                        elif state == 2:
                            CAM.start()
                            videoCapture = True
                        continue
                    elif event.key == K_UP:
                        cursorMoveTo = -1
                    elif event.key == K_DOWN:
                        cursorMoveTo = +1
                    elif event.key == K_ESCAPE:
                        terminate()

            if cursorMoveTo != 0:
                menuNeedsRedraw = True

        elif uploadedPhoto == True:
            DISPLAYSURF.fill(MAPCOLORING.WHITE)

            pleaseWaitSurf = pygame.Surface((MAPCOLORING.WINWIDTH, MAPCOLORING.WINHEIGHT))
            pleaseWaitSurf.fill(MAPCOLORING.BLUE)

            runningTxt, runningTxtRect = drawText('Image is being processing', 0, 0)
            runningTxtRect.center = ((MAPCOLORING.WINWIDTH/2, MAPCOLORING.WINHEIGHT/2))
            pleaseWaitSurf.blit(runningTxt, runningTxtRect)

            pleaseWaitTxt, pleaseWaitTxtRect = drawText('Please wait a few seconds', 0, 0)
            pleaseWaitTxtRect.center =((MAPCOLORING.WINWIDTH/2, MAPCOLORING.WINHEIGHT/2 + 30))
            pleaseWaitSurf.blit(pleaseWaitTxt, pleaseWaitTxtRect)

            pleaseWaitSurfRct = pleaseWaitSurf.get_rect()
            pleaseWaitSurfRct.topleft = (0,0)
            DISPLAYSURF.blit(pleaseWaitSurf, pleaseWaitSurfRct)

            pygame.display.update()
            
            t0 = pygame.time.get_ticks()

            
            output = detect_image(image, model, face_cascade)
            
            runtime = pygame.time.get_ticks() - t0
            forTransformImage = deepcopy(output)
            map = cv2ImageToSurface(forTransformImage)
            menuNeedsRedraw = True
            mapNeedsRedraw = True
            
            uploadedPhoto = False

        if videoCapture is True:
            if CAM.query_image():
                image = surfaceToCv2(CAM.get_image())
                t0 = pygame.time.get_ticks()
                output = detect_image(image, model, face_cascade)
                runtime = pygame.time.get_ticks() - t0
                forTransformImage = deepcopy(output)
                map = cv2ImageToSurface(forTransformImage)
                mapNeedsRedraw = True


        DISPLAYSURF.fill(MAPCOLORING.WHITE)

        if menuNeedsRedraw:
            state = state + cursorMoveTo
            if state < 0:
                state = MAXSTATE
            elif state > MAXSTATE:
                state = 0
            menuSurf = drawMenu(state, runtime)
            menuNeedsRedraw = False

        menuSurfRect = menuSurf.get_rect()
        menuSurfRect.topleft = (0,0)
        DISPLAYSURF.blit(menuSurf, menuSurfRect)

        if mapNeedsRedraw:
            mapSurf = drawMap(map)
            mapNeedsRedraw = False
        mapSurfRect = mapSurf.get_rect()
        mapSurfRect.topleft = (MAPCOLORING.WINWIDTH*9/25,0)
        DISPLAYSURF.blit(mapSurf, mapSurfRect)
        
        pygame.display.update()

def uploadPhoto():
    file_path = filedialog.askopenfilename(filetypes=[('Image Files', ('.png', '.jpg'))])
    if file_path:
        image = cv2.imread(file_path,1)
        return image
    else:
        return None

def savePhoto(image):
    file = filedialog.asksaveasfile(mode='w', defaultextension=".png")
    if file:
        cv2.imwrite(file.name, image)
        return True
    return False

def imgScale(image):
    width = image.get_width()
    height = image.get_height()
    scaleW = 1
    scaleH = 1
    if width > MAPCOLORING.WINWIDTH * (16/25):
        scaleW = MAPCOLORING.WINWIDTH * (16 / 25) / width
    if height > MAPCOLORING.WINHEIGHT:
        scaleH = MAPCOLORING.WINHEIGHT / height 
    scale = 1
    if scaleW > scaleH:
        scale = scaleH
    else:
        scale = scaleW
    
    scaledImg =  pygame.transform.scale(image, (int(width * scale), int(height * scale)))
    return scaledImg

def drawMap(map):
    image = imgScale(map)
    mapSurf = pygame.Surface((MAPCOLORING.WINWIDTH* (16/25), MAPCOLORING.WINHEIGHT))
    mapSurf.fill(MAPCOLORING.GREY)
    mapRect = image.get_rect()
    mapRect.center = (MAPCOLORING.WINWIDTH * (8 / 25) , MAPCOLORING.HALF_WINHEIGHT)
    mapSurf.blit(image, mapRect)
    return mapSurf

def drawMenu(state, runtime):
    menuSurf = pygame.Surface((MAPCOLORING.WINWIDTH * 9/25, MAPCOLORING.WINHEIGHT))
    menuSurf.fill(MAPCOLORING.BGCOLOR) 
    nameSurface, nameRect = drawText('Menu', 120, 250)
    menuSurf.blit(nameSurface, nameRect)

    xtop, ytop = 45, 300
    for x in range(len(BUTTONS)):
        btnSurface, btnRect = drawText(BUTTONS[x], xtop, ytop + 25 * x)
        menuSurf.blit(btnSurface, btnRect)

    cursorSurface, cursorRect = drawText('*', xtop - 25, ytop + 25 * state)
    menuSurf.blit(cursorSurface, cursorRect)

    runtimeSurface, runtimeRect = drawRuntime('Runtime: ' + str(round(runtime/1000,5)) + ' s', 55, 500)
    menuSurf.blit(runtimeSurface, runtimeRect)

    return menuSurf

def drawRuntime(text, x, y):
    textSurface = RUNTIMEFONT.render(text, True, MAPCOLORING.TEXTCOLOR)
    textRect = textSurface.get_rect()
    textRect.topleft = (x, y)
    return textSurface, textRect

def drawText(text, x, y):
    textSurface = BASICFONT.render(text, True, MAPCOLORING.TEXTCOLOR)
    textRect = textSurface.get_rect()
    textRect.topleft = (x, y)
    return textSurface, textRect

def cv2ImageToSurface(cv2Image):
    if cv2Image.dtype.name == 'uint16':
        cv2Image = (cv2Image / 256).astype('uint8')
    size = cv2Image.shape[1::-1]
    if len(cv2Image.shape) == 2:
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
        format = 'RGB'
    else:
        format = 'RGBA' if cv2Image.shape[2] == 4 else 'RGB'
        cv2Image[:, :, [0, 2]] = cv2Image[:, :, [2, 0]]
    surface = pygame.image.frombuffer(cv2Image.flatten(), size, format)
    return surface.convert_alpha() if format == 'RGBA' else surface.convert()

def surfaceToCv2(surface):
    #  create a copy of the surface
    view = pygame.surfarray.array3d(surface)

    #  convert from (width, height, channel) to (height, width, channel)
    view = view.transpose([1, 0, 2])

    #  convert from rgb to bgr
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

    return img_bgr

def terminate():
    pygame.quit()
    sys.exit()
    
if __name__ == '__main__':
    main()