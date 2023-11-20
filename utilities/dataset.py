import queue
import threading
import time
import traceback

import cv2


class CVFreshestFrame:
    """
    always getting the most recent frame of a camera

    """

    def __init__(self, source, fps=30):
        self.cap = cv2.VideoCapture(source)
        self.running = True
        self.fps = fps
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def release(self):
        self.running = False
        self.cap.release()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                self.release()
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            time.sleep(1 / self.fps)

    def read(self):
        # print("read", (self.q.get() if self.running and not self.q.empty() else None) is None)
        return self.running, self.q.get()


if __name__ == "__main__":
    pass
