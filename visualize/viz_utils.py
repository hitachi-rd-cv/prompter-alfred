import os
import re
import cv2

import numpy as np


class Vid:
    def __init__(self, vidname, fps, dim, elems):
        self.w, self.h = dim
        self.elems = elems
        self.last_canvas = np.zeros((self.w, self.h, 3), dtype=np.uint8)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.video = cv2.VideoWriter(vidname, fourcc, fps, dim)

    def getMaxFrame(self):
        return max([elem.getMaxFrame() for elem in self.elems])

    def draw(self, frame_n):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        for elem in self.elems:
            canvas = elem.draw(canvas, frame_n)

        self.video.write(canvas)
        self.last_canvas = canvas

    def release(self):
        for _ in range(10):
            self.video.write(self.last_canvas)
        self.video.release()


def extractFrames(dname):
    imgnames = os.listdir(dname)
    frames = [int(re.findall("[0-9]+", name)[0]) for name in imgnames]
    return sorted(frames)


def placeImg(canvas, img, atx=0, aty=0):
    h, w = img.shape[:2]
    canvas[aty:aty+h, atx:atx+w] = img
    return canvas


class ImgElem:
    def __init__(self, dirname, fname_template, pos, text=None, do_flip=False, crop=None, resize=None, heatmap=False):
        self.dirname = dirname
        self.frames = extractFrames(self.dirname)
        self.fname_template = fname_template
        self.pos = pos
        self.text = text
        self.do_flip = do_flip
        self.crop = crop
        self.resize = resize
        self.heatmap = heatmap

        if resize is None:
            self.h, self.w = cv2.imread(os.path.join(self.dirname, os.listdir(self.dirname)[0])).shape[:2]
        else:
            self.w, self.h = resize
        self.curr_img = np.zeros((self.w, self.h, 3), dtype=np.uint8)

    def draw(self, canvas, frame_n):
        if frame_n in self.frames:
            self.curr_img = cv2.imread(
                os.path.join(self.dirname, self.fname_template % frame_n))
            if self.crop is not None:
                pt1, pt2 = self.crop[0], self.crop[1]
                self.curr_img = self.curr_img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            if self.do_flip:
                self.curr_img = cv2.flip(self.curr_img, 0)
            if self.resize is not None:
                self.curr_img = cv2.resize(self.curr_img, self.resize)
            if self.heatmap:
                self.curr_img[self.curr_img > 10] = 160
                self.curr_img = cv2.applyColorMap(self.curr_img, cv2.COLORMAP_JET)
        return self.placeText(self.placeImg(canvas))

    def placeImg(self, canvas):
        return placeImg(canvas, self.curr_img, self.pos[0], self.pos[1])

    def placeText(self, canvas):
        return cv2.putText(
            canvas, self.text, (self.pos[0], self.pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    def getMaxFrame(self):
        return max(self.frames)


class TxtElem:
    def __init__(self, pos, size=1, thickness=2, color=(255, 255, 255), contents=None, newline_down=True):
        self.pos = pos
        self.size = size
        self.thickness = thickness
        self.color = color
        self.newline_down = newline_down
        self.newline_sz = (30 * self.size) * (1 if newline_down else -1)
        self.update(contents)

    def update(self, contents):
        if type(contents) != list:
            contents = [contents]
        if not self.newline_down:
            contents.reverse()
        self.contents = contents

    def draw(self, canvas, frame_n):
        if self.contents is None:
            return canvas

        x, y = self.pos
        for indx, content in enumerate(self.contents):
            canvas = cv2.putText(
                canvas, content, (x, int(y + self.newline_sz * (indx + 1))),
                cv2.FONT_HERSHEY_SIMPLEX, self.size, self.color,
                self.thickness, cv2.LINE_AA)
        return canvas

    def getMaxFrame(self):
        return 0
