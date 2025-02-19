#coding=utf-8
from PIL import Image,ImageDraw,ImageFont
import PIL
import cv2
import numpy as np

def get_text_rect(img,pos,text,font,margin0=7,margin1=7):
    img = Image.fromarray(img)


    painter = ImageDraw.Draw(img)
    text_w = 1
    text_h = 1
    for t in text:
        tw,th= painter.textsize(t,font=font)
        text_h = th
        text_w = max(text_w,tw)

    rect = [pos[0]-margin0,pos[1]-margin0,pos[0]+text_w+margin0,pos[1]+text_h+(text_h+margin1)*(len(text)-1)+2*margin0]
    rect = [max(x,0) for x in rect]
    return rect,text_w,text_h

def __get_img_rect(img,rect):
    return img[rect[1]:rect[3],rect[0]:rect[2]]

def draw_text(img,pos,text,background_color=(0,0,255),text_color=(255,255,255),font_size=17,
              font=None,margin0=7,margin1=7,alpha=0.4):
    '''

    Args:
        img: [H,W,3] rgb image
        pos: (x,y) draw position
        text: a str a list of str text do draw
        background_color: rgb rect background color or None
        text_color: rgb text color
        font_size:
        font: ImageFont.trutype or font path
        margin0: text top left right bottom margin
        margin1: margin between text line
        alpha: background alpha

    Returns:

    '''
    if isinstance(text,str):
        text = [text]
    if isinstance(font,str):
        font = ImageFont.truetype(font, font_size)
    rect,text_w,text_h = get_text_rect(img,pos, text,
                                       font,
                                       margin0=margin0,
                                       margin1=margin1)
    if background_color is not None:
        target_img = __get_img_rect(img,rect)
        background = np.array(list(background_color)).reshape([1,1,3])*alpha
        target_img = (target_img*(1-alpha)+background).astype(np.uint8)
        img[rect[1]:rect[3],rect[0]:rect[2]] = target_img

    img = Image.fromarray(img)
    painter = ImageDraw.Draw(img)
    y = pos[1]
    for t in text:
        painter.text((pos[0],y),t,
                     font=font,
                     fill=text_color)
        y += text_h+margin1

    return np.array(img)

if __name__ == "__main__":
    img = cv2.imread("/home/wj/ai/work/semantic-segmentation/example_imgs/28527aac14d29e67d301cf9b636800b7.jpeg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = draw_text(img,(20,20),["text0","textext  hello world.","中文测试"],
                        font='/home/wj/ai/file/font/simhei.ttf')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow("test",img)
    cv2.waitKey(-1)