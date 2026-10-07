"""Non-blocking offline text-to-speech."""
import queue
import threading


class Speaker:
    def __init__(self):
        self.q = queue.Queue()
        try:
            import pyttsx3  # noqa
            self.ok = True
            threading.Thread(target=self._run, daemon=True).start()
        except Exception:
            self.ok = False

    def _run(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        while True:
            engine.say(self.q.get())
            engine.runAndWait()

    def say(self, text):
        if self.ok and text:
            self.q.put(text)
