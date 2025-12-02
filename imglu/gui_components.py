import cv2
import time

class Video():

    def __init__(
        self, 
        path: str, 
        loop: bool = False):

        self.path = path

        self.video_path = path 
        self.video = cv2.VideoCapture(path)

        self.loop = loop

        self.last_frame_time = None

        self.frame_index = 0
        self.frame = cv2.cvtColor(self.video.read()[1], cv2.COLOR_BGR2RGB)
        self.f = 0

        self.play_toggle = False

        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.n_frames  = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration  = self.n_frames/self.video_fps

    def get_frame(self):

        if self.video is None or \
           not self.play_toggle or \
           (not self.loop and self.frame_index == self.n_frames):
            return self.frame

        d = (time.time() - self.last_frame_time)*self.video_fps
        if d >= 1.0:
            self.last_frame_time = time.time()
            self.f += d

        if self.frame_index <= self.n_frames:

            while 1.0 <= self.f:

                flag, frame = self.video.read()
                if flag:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_index += 1
                self.f = self.f - 1.0

            return self.frame
        else:
            if self.loop:

                self.last_frame_time = time.time()
                self.frame_index = 0
                self.f = 0

                self.video.release()
                self.video = cv2.VideoCapture(self.video_path)

                return self.frame
            else:
                return self.frame

    def reset_and_play(self):

        self.reset()
        self.play()

    def pause(self):

        self.play_toggle = False

    def resume(self):

        self.play_toggle = True
        self.last_frame_time = time.time()
        self.f          = 0

    def play(self):

        self.play_toggle = True
        self.last_frame_time = time.time()
        self.frame_index = 0
        self.f = 0

    def reset(self):

        self.video.release()
        self.video = cv2.VideoCapture(self.video_path)

    def __del__(self):
        self.video.release()
