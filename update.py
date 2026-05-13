"""
AirScroll — Gesture + Voice Controller
Team   : Vaibhav Kaushik (CB8149), Aashu Dewangan (CB8148), Agash Kumar (CB8168)
Guide  : Mrs. Surabhi Parekh — Asst. Prof. (I.T.)
College: CIT Jagdalpur, Bastar, Chhattisgarh  |  Phase 2

GESTURE SYSTEM (palm only):
  🖐 Palm swipe LEFT  → ← Arrow key (Back / Previous)
  🖐 Palm swipe RIGHT → → Arrow key (Next / Forward)
  🖐 Palm swipe UP    → ↑ Arrow key (Up)
  🖐 Palm swipe DOWN  → ↓ Arrow key (Down)

VOICE: Runs simultaneously — speak any command.
"""

import os, sys, time, threading, datetime, webbrowser, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]      = "2"

import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

# ── Optional libs ─────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PIL import ImageGrab
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PAG_OK = True
except ImportError:
    PAG_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except ImportError:
    SR_OK = False

try:
    import ctypes
    CTYPES_OK = True
except ImportError:
    CTYPES_OK = False

try:
    from flask import Flask, request, jsonify, Response
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

try:
    import qrcode
    QR_OK = True
except ImportError:
    QR_OK = False

# ── Selenium (optional — for DOM-based ad detection) ──────────────
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (NoSuchElementException,
                                             TimeoutException,
                                             WebDriverException)
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False

# ══════════════════════════════════════════════════════════════════
#  COLORS
# ══════════════════════════════════════════════════════════════════
BG     = "#07071A"
BG2    = "#0C0C24"
CARD   = "#10102A"
CARD2  = "#161636"
BDR    = "#1E1E45"
GOLD   = "#F5C400"
CYAN   = "#00DCFF"
GREEN  = "#00F573"
RED    = "#FF3D5A"
PURPLE = "#A855F7"
BLUE   = "#3B82F6"
ORANGE = "#F97316"
TEXT   = "#E2E2F5"
MUTED  = "#4A4A7A"
MUTED2 = "#6A6A9A"

# ══════════════════════════════════════════════════════════════════
#  GESTURE CLASSIFIER  — palm swipes only
# ══════════════════════════════════════════════════════════════════
def _fingers_up(lm):
    """Returns (thumb, index, middle, ring, pinky) as 1/0."""
    lm = lm.landmark
    th = 1 if lm[4].x < lm[3].x else 0
    return (th,
            1 if lm[8].y  < lm[6].y  else 0,
            1 if lm[12].y < lm[10].y else 0,
            1 if lm[16].y < lm[14].y else 0,
            1 if lm[20].y < lm[18].y else 0)


class PalmClassifier:
    """
    Detects palm swipe directions.
    Returns: PALM_UP / PALM_DOWN / PALM_LEFT / PALM_RIGHT / PALM / IDLE

    One gesture fires per palm raise. Palm must close (IDLE) between
    swipes so the engine can count them cleanly.

    Thresholds are generous — detects any clear directional movement.
    """
    SWIPE_H    = 0.10   # horizontal fraction of frame
    SWIPE_V    = 0.09   # vertical fraction of frame
    MIN_FRAMES = 3      # wait this many frames before detecting

    def __init__(self):
        self._sx    = None   # wrist X when palm opened
        self._sy    = None   # wrist Y when palm opened
        self._n     = 0      # consecutive palm frames
        self._fired = False  # fired once this palm raise

    def classify(self, lm_obj):
        try:    return self._do(lm_obj)
        except: return "IDLE"

    def _do(self, lm_obj):
        th, ix, mi, ri, pi = _fingers_up(lm_obj)
        lm = lm_obj.landmark
        wx = lm[0].x
        wy = lm[0].y

        # Not a full open palm → reset everything
        if not (th and ix and mi and ri and pi):
            self._sx = None; self._sy = None
            self._n  = 0;    self._fired = False
            return "IDLE"

        # Palm just opened → record anchor
        if self._sx is None:
            self._sx = wx; self._sy = wy; self._n = 1
            self._fired = False
            return "PALM"

        self._n += 1

        # Measure displacement from anchor
        dx = wx - self._sx
        dy = wy - self._sy

        # Need minimum frames before firing
        if self._n < self.MIN_FRAMES:
            return "PALM"

        # Fire once per palm raise
        if not self._fired:
            if abs(dy) >= abs(dx):               # vertical dominant
                if   dy < -self.SWIPE_V:
                    self._fired = True; return "PALM_UP"
                elif dy >  self.SWIPE_V:
                    self._fired = True; return "PALM_DOWN"
            else:                                 # horizontal dominant
                if   dx < -self.SWIPE_H:
                    self._fired = True; return "PALM_LEFT"
                elif dx >  self.SWIPE_H:
                    self._fired = True; return "PALM_RIGHT"

        return "PALM"


class ThreeFingerScroll:
    """
    Detects thumb + index + middle fingers up (ring + pinky down).
    While held, tracks wrist Y movement each frame:
      - Moving UP   → fires SCROLL_UP  every frame
      - Moving DOWN → fires SCROLL_DOWN every frame
      - Still       → returns THREE (neutral, no scroll)

    This gives smooth continuous scrolling — as long as you keep
    moving your hand up/down, scroll fires every frame.
    """
    MOVE_T = 0.001   # ultra sensitive — fires on slightest movement

    def __init__(self):
        self._pwy = None   # previous frame wrist Y
        self._n   = 0      # consecutive three-finger frames

    def classify(self, lm_obj):
        try:    return self._do(lm_obj)
        except: return "IDLE"

    def _do(self, lm_obj):
        th, ix, mi, ri, pi = _fingers_up(lm_obj)
        lm = lm_obj.landmark
        wy = lm[0].y

        if not (th and ix and mi and not ri and not pi):
            self._pwy = None; self._n = 0
            return "IDLE"

        pwy = self._pwy
        self._pwy = wy
        self._n  += 1

        if pwy is None or self._n < 3:
            return "THREE"

        dy = wy - pwy

        if dy < -self.MOVE_T:
            return "SCROLL_UP"
        if dy >  self.MOVE_T:
            return "SCROLL_DOWN"

        return "THREE"

class DoublePinchDetector:
    """Two quick pinches within 0.55s → DOUBLE_PINCH."""
    PINCH_T    = 0.07
    DOUBLE_WIN = 0.55

    def __init__(self):
        self._last_t   = 0.0
        self._in_pinch = False

    def classify(self, lm_obj):
        try:    return self._do(lm_obj)
        except: return "IDLE"

    def _do(self, lm_obj):
        now = time.time()
        pd  = pinch_dist(lm_obj)
        if pd < self.PINCH_T:
            if not self._in_pinch:
                self._in_pinch = True
                gap = now - self._last_t
                self._last_t = now
                if gap < self.DOUBLE_WIN:
                    return "DOUBLE_PINCH"
                return "PINCH"
        else:
            self._in_pinch = False
        return "IDLE"


# ══════════════════════════════════════════════════════════════════
#  VOICE COMMANDS TABLE + EXECUTOR
# ══════════════════════════════════════════════════════════════════
VOICE_COMMANDS = {
    # Apps
    "open youtube":"youtube",      "youtube":"youtube",
    "youtube kholo":"youtube",     "youtube chalu karo":"youtube",
    "youtube chalu kar":"youtube", "youtube chalu kar do":"youtube",
    "open chrome":"chrome",        "chrome":"chrome",
    "open excel":"excel",          "excel":"excel",
    "open word":"word",            "word":"word",
    "open notepad":"notepad",      "notepad":"notepad",
    "my computer":"mycomputer",    "file explorer":"mycomputer",
    "open downloads":"downloads",  "open documents":"documents",
    "open desktop":"desktop",
    # Mirror panel
    "show mirror":"show_mirror",   "mirror on":"show_mirror",
    "mirror dikhao":"show_mirror", "mirror kholo":"show_mirror",
    "mirror panel dikhao":"show_mirror",
    "hide mirror":"hide_mirror",   "mirror off":"hide_mirror",
    "remove mirror":"hide_mirror", "close mirror":"hide_mirror",
    "mirror band":"hide_mirror",   "mirror band karo":"hide_mirror",
    "mirror hatao":"hide_mirror",  "mirror panel hatao":"hide_mirror",
    # System
    "screenshot":"screenshot",     "take screenshot":"screenshot",
    "lock screen":"lockscreen",    "lock":"lockscreen",
    "task manager":"taskmanager",
    "close all":"closeall",
    "show desktop":"show_desktop",
    "minimize":"minimize",         "close window":"close_window",
    # Volume
    "volume up":"volumeup",        "louder":"volumeup",
    "volume down":"volumedown",    "quieter":"volumedown",
    "mute":"mute",                 "unmute":"mute",
    # Media
    "play":"playpause",            "pause":"playpause",
    "play pause":"playpause",
    "next track":"nexttrack",      "next":"nexttrack",
    "previous track":"prevtrack",  "previous":"prevtrack",
    # Navigation
    "go back":"go_back",           "back":"go_back",
    "go forward":"go_forward",     "forward":"go_forward",
    "scroll up":"scroll_up",       "scroll down":"scroll_down",
    # Keys
    "press enter":"key_enter",     "enter":"key_enter",
    "press up":"key_up",           "press down":"key_down",
    "press left":"key_left",       "press right":"key_right",
    # Edit
    "copy":"copy",                 "paste":"paste",
    "cut":"cut",                   "undo":"undo",
    "redo":"redo",                 "save":"save",
    "select all":"select_all",     "zoom in":"zoom_in",
    "zoom out":"zoom_out",
}


def _exec_voice_action(cmd):
    """Execute a voice command by its command ID. Returns status string."""
    import subprocess, os

    if cmd == "youtube":
        # Launch Chrome with remote debugging so AdSkipper can attach
        try:
            import subprocess as _sp
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Users\{}\AppData\Local\Google\Chrome\Application\chrome.exe".format(
                    os.environ.get("USERNAME","")),
            ]
            launched = False
            for path in chrome_paths:
                if os.path.exists(path):
                    _sp.Popen([path,
                               "--remote-debugging-port=9222",
                               "--no-first-run",
                               "https://www.youtube.com"])
                    launched = True
                    break
            if not launched:
                webbrowser.open("https://www.youtube.com")
        except Exception:
            webbrowser.open("https://www.youtube.com")
        return "Opened YouTube"
    if cmd == "chrome":
        try: subprocess.Popen(["start","chrome"], shell=True)
        except Exception: webbrowser.open("https://www.google.com")
        return "Opened Chrome"
    if cmd == "excel":
        try: subprocess.Popen(["start","excel"], shell=True)
        except Exception: pass
        return "Opening Excel"
    if cmd == "word":
        try: subprocess.Popen(["start","winword"], shell=True)
        except Exception: pass
        return "Opening Word"
    if cmd == "notepad":
        try: subprocess.Popen(["notepad.exe"])
        except Exception: pass
        return "Opening Notepad"
    if cmd == "mycomputer":
        try: subprocess.Popen(["explorer.exe","::{20D04FE0-3AEA-1069-A2D8-08002B30309D}"])
        except Exception: pass
        return "Opening My Computer"
    if cmd == "documents":
        try: subprocess.Popen(["explorer.exe", os.path.expanduser("~\\Documents")])
        except Exception: pass
        return "Opening Documents"
    if cmd == "downloads":
        try: subprocess.Popen(["explorer.exe", os.path.expanduser("~\\Downloads")])
        except Exception: pass
        return "Opening Downloads"
    if cmd == "desktop":
        try: subprocess.Popen(["explorer.exe", os.path.expanduser("~\\Desktop")])
        except Exception: pass
        return "Opening Desktop"
    if cmd == "screenshot":
        if PAG_OK:
            try: pyautogui.hotkey("win","shift","s")
            except Exception: pass
        return "Screenshot"
    if cmd == "lockscreen":
        if CTYPES_OK:
            try: ctypes.windll.user32.LockWorkStation()
            except Exception: pass
        return "Lock screen"
    if cmd == "taskmanager":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","shift","esc")
            except Exception: pass
        return "Task Manager"
    if cmd == "closeall":
        if PAG_OK:
            try: pyautogui.hotkey("alt","f4")
            except Exception: pass
        return "Close all"
    if cmd in ("show_desktop","show desktop"):
        if PAG_OK:
            try: pyautogui.hotkey("win","d")
            except Exception: pass
        return "Show desktop"
    if cmd == "minimize":
        if PAG_OK:
            try: pyautogui.hotkey("win","down")
            except Exception: pass
        return "Minimize"
    if cmd == "close_window":
        if PAG_OK:
            try: pyautogui.hotkey("alt","f4")
            except Exception: pass
        return "Close window"
    if cmd == "volumeup":
        if PAG_OK:
            try:
                for _ in range(3): pyautogui.press("volumeup")
            except Exception: pass
        return "Volume up"
    if cmd == "volumedown":
        if PAG_OK:
            try:
                for _ in range(3): pyautogui.press("volumedown")
            except Exception: pass
        return "Volume down"
    if cmd == "mute":
        if PAG_OK:
            try: pyautogui.press("volumemute")
            except Exception: pass
        return "Mute toggle"
    if cmd == "playpause":
        if PAG_OK:
            try: pyautogui.press("playpause")
            except Exception: pass
        return "Play/Pause"
    if cmd == "nexttrack":
        if PAG_OK:
            try: pyautogui.press("nexttrack")
            except Exception: pass
        return "Next track"
    if cmd == "prevtrack":
        if PAG_OK:
            try: pyautogui.press("prevtrack")
            except Exception: pass
        return "Prev track"
    if cmd == "go_back":
        if PAG_OK:
            try: pyautogui.hotkey("alt","left")
            except Exception: pass
        return "Go back"
    if cmd == "go_forward":
        if PAG_OK:
            try: pyautogui.hotkey("alt","right")
            except Exception: pass
        return "Go forward"
    if cmd == "scroll_up":
        if PAG_OK:
            try: pyautogui.scroll(30)
            except Exception: pass
        return "Scroll up"
    if cmd == "scroll_down":
        if PAG_OK:
            try: pyautogui.scroll(-30)
            except Exception: pass
        return "Scroll down"
    if cmd == "key_enter":
        if PAG_OK:
            try: pyautogui.press("enter")
            except Exception: pass
        return "Enter"
    if cmd == "key_up":
        if PAG_OK:
            try: pyautogui.press("up")
            except Exception: pass
        return "↑"
    if cmd == "key_down":
        if PAG_OK:
            try: pyautogui.press("down")
            except Exception: pass
        return "↓"
    if cmd == "key_left":
        if PAG_OK:
            try: pyautogui.press("left")
            except Exception: pass
        return "←"
    if cmd == "key_right":
        if PAG_OK:
            try: pyautogui.press("right")
            except Exception: pass
        return "→"
    if cmd == "copy":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","c")
            except Exception: pass
        return "Copy"
    if cmd == "paste":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","v")
            except Exception: pass
        return "Paste"
    if cmd == "cut":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","x")
            except Exception: pass
        return "Cut"
    if cmd == "undo":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","z")
            except Exception: pass
        return "Undo"
    if cmd == "redo":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","y")
            except Exception: pass
        return "Redo"
    if cmd == "save":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","s")
            except Exception: pass
        return "Save"
    if cmd == "select_all":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","a")
            except Exception: pass
        return "Select all"
    if cmd == "zoom_in":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","=")
            except Exception: pass
        return "Zoom in"
    if cmd == "zoom_out":
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","-")
            except Exception: pass
        return "Zoom out"
    if cmd == "show_mirror":
        return "show_mirror_signal"
    if cmd == "hide_mirror":
        return "hide_mirror_signal"
    return f"Unknown: {cmd}"


class VoiceEngine:
    """
    Voice-controlled file navigator + command executor.

    NAVIGATION SESSION:
      User says "local disk d" / "documents" / "downloads" etc.
      → Engine opens that folder and lists its contents
      → User says a name or number → navigates into it
      → Continues until user opens a file or says "back"/"cancel"

    COMMANDS work at any time alongside navigation.
    """

    # ── Known root locations ──────────────────────────────────────
    ROOT_LOCATIONS = {
        # English
        "my computer":      "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}",
        "computer":         "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}",
        "this pc":          "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}",
        "local disk c":     "C:\\",
        "c drive":          "C:\\",
        "disk c":           "C:\\",
        "drive c":          "C:\\",
        "local disk d":     "D:\\",
        "d drive":          "D:\\",
        "disk d":           "D:\\",
        "drive d":          "D:\\",
        "local disk e":     "E:\\",
        "e drive":          "E:\\",
        "disk e":           "E:\\",
        "documents":        os.path.expanduser("~\\Documents"),
        "my documents":     os.path.expanduser("~\\Documents"),
        "downloads":        os.path.expanduser("~\\Downloads"),
        "desktop":          os.path.expanduser("~\\Desktop"),
        "pictures":         os.path.expanduser("~\\Pictures"),
        "music":            os.path.expanduser("~\\Music"),
        "videos":           os.path.expanduser("~\\Videos"),
        # Hindi / Hinglish variants
        "mera computer":    "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}",
        "local disk di":    "D:\\",
        "d ki disk":        "D:\\",
        "d wali disk":      "D:\\",
        "documents folder": os.path.expanduser("~\\Documents"),
        "downloads folder": os.path.expanduser("~\\Downloads"),
    }

    def __init__(self, log_fn=None, api_key=""):
        self.log_fn         = log_fn or print
        self._api_key       = api_key
        self._running       = False
        self._rec           = None
        self._mic           = None
        self._nav_path      = None
        self._nav_items     = []
        self._nav_history   = []
        self._browser_ctx   = None
        self._mic_paused    = False
        self._WAKE_WORDS    = ["pause","rok","roko","stop","sun","suno",
                                "wait","thehro","ruk","volume","fullscreen",
                                "next","skip","mute","niche","upar"]
        self._yt_results    = []
        self._mirror       = None
        self._PLAY_WORDS    = ["bajao","chala","chalao","laga","play karo",
                                "play kar","play","search karo","dhundo",
                                "suno","sun","dekho","chalu karo","chalu kar",
                                "shuru karo","on kar","dede","dikhao","lagao",
                                "start","run karo","bja","bja do"]
        self._FILLER_WORDS  = ["pe","par","mein","ko","ka","ki","ke",
                                "please","zara","bhai","yaar","jaldi",
                                "abhi","wala","wali","yeh","ye","voh","vo",
                                "mujhe","ek","aur","toh"]

    def start(self):
        if not SR_OK:
            self.log_fn("⚠ Voice: pip install SpeechRecognition pyaudio")
            return False
        if self._running:
            return True
        try:
            self._rec = sr.Recognizer()
            self._rec.energy_threshold         = 300
            self._rec.dynamic_energy_threshold = True
            self._rec.pause_threshold          = 0.6
            self._mic = sr.Microphone()
            with self._mic as src:
                self._rec.adjust_for_ambient_noise(src, duration=1)
            self._running = True
            threading.Thread(target=self._loop, daemon=True).start()
            self.log_fn("🎤 Voice started — speak a command!")
            return True
        except Exception as e:
            self.log_fn(f"⚠ Mic error: {e}"); return False

    def stop(self):
        self._running = False
        self.log_fn("🎤 Voice stopped")

    def _loop(self):
        fail = 0
        while self._running:

            # ── MIC PAUSED — zero audio capture ───────────────────
            if self._mic_paused:
                time.sleep(0.1)
                continue

            # ── NORMAL listening ──────────────────────────────────
            try:
                with self._mic as src:
                    # Short timeout = loop checks _mic_paused every 0.5s max
                    audio = self._rec.listen(src, timeout=0.5, phrase_time_limit=6)

                if self._mic_paused:
                    continue   # paused mid-capture — discard

                text = self._rec.recognize_google(audio).lower().strip()

                if self._mic_paused or not text:
                    continue   # final guard

                fail = 0
                self.log_fn(f"🎤 \"{text}\"")
                self._handle(text)

            except sr.WaitTimeoutError:
                continue   # silence — loop back and re-check _mic_paused
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                self.log_fn(f"⚠ API: {e}"); fail += 1
                if fail > 5: self.log_fn("⚠ Check internet"); fail = 0
                time.sleep(2)
            except OSError:
                self.log_fn("⚠ Mic lost — reconnecting...")
                time.sleep(2)
                try:
                    self._mic = sr.Microphone()
                    with self._mic as src:
                        self._rec.adjust_for_ambient_noise(src, duration=0.5)
                    self.log_fn("🎤 Mic reconnected")
                except Exception: pass
            except Exception as e:
                if self._running: self.log_fn(f"⚠ Voice error: {e}")
                time.sleep(0.5)

    def pause_mic(self):
        """Completely stop mic capture. Called when video starts playing."""
        if not self._mic_paused:
            self._mic_paused = True
            self.log_fn("🎤 Mic OFF — show palm or say a wake word to resume")

    def resume_mic(self):
        """Resume mic capture after a short settling delay."""
        def _resume():
            time.sleep(0.8)   # wait for video audio to fade from mic
            self._mic_paused = False
            self.log_fn("🎤 Mic ON — listening")
        threading.Thread(target=_resume, daemon=True).start()

    def _handle(self, text):
        t = text.strip().lower()

        # ── Global browser close commands (fire regardless of context) ─
        CLOSE_TAB_G = ["close this tab","close tab","tab band karo","tab band",
                       "tab close","close current tab","is tab ko band karo",
                       "yeh tab band","tab close karo","tab close kro",
                       "close it","band karo tab","isko band karo"]
        CLOSE_BR_G  = ["close browser","browser band","browser close",
                       "browser ko band karo","close window","band karo browser"]
        if any(w in t for w in CLOSE_TAB_G):
            self._close_current_tab()
            self.log_fn("🗂 Tab closed"); return
        if any(w in t for w in CLOSE_BR_G):
            self._close_browser()
            self.log_fn("🌐 Browser closed"); return

        # ── Navigation session active ─────────────────────────────
        if self._nav_path is not None:
            self._nav_input(t)
            return

        # ── Navigation trigger always wins over browser context ──────
        nav_path = self._detect_location(t)
        if nav_path:
            self._browser_ctx = None
            self._nav_enter(nav_path)
            return

        # ── Browser context — smart in-app commands ───────────────
        if self._browser_ctx:
            if self._browser_handle(t):
                return

        # ── Dynamic web search ────────────────────────────────────
        for trigger in ("search ","google ","search for ","search karo "):
            if t.startswith(trigger):
                q = t[len(trigger):].strip()
                if not q: continue
                # Use Selenium if driver available → shows results in panel
                driver = self._get_driver() if hasattr(self,'_get_driver') else None
                webbrowser.open(f"https://www.google.com/search?q={q.replace(' ','+')}")
                self.log_fn(f"🔍 Search: \"{q}\"")
                return

        if "go to" in t or "open website" in t:
            words = t.replace("go to","").replace("open website","").strip()
            if words:
                url = f"https://{words}" if not words.startswith("http") else words
                webbrowser.open(url); self.log_fn(f"🌐 {url}"); return

        # ── Local command match ───────────────────────────────────
        best_phrase, best_sid = "", None
        for phrase, sid in VOICE_COMMANDS.items():
            if (phrase == t or phrase in t) and len(phrase) > len(best_phrase):
                best_phrase, best_sid = phrase, sid
        if best_sid:
            # Mirror signals — pass up to UI via log
            if best_sid == "show_mirror":
                msg = _exec_voice_action("show_mirror")
                self.log_fn(f"✓ {msg}"); return
            if best_sid == "hide_mirror":
                msg = _exec_voice_action("hide_mirror")
                self.log_fn(f"✓ {msg}"); return
            # Set browser context when opening sites
            if best_sid == "youtube":
                self._browser_ctx = "youtube"
                self.log_fn("📺 YouTube context active — say song name to play")
                # Connect ad skipper in background
                threading.Thread(
                    target=lambda: _ensure_ad_skipper_connected(self.log_fn),
                    daemon=True).start()
            elif best_sid == "chrome":
                self._browser_ctx = "chrome"
            elif best_sid == "google":
                self._browser_ctx = "google"
            msg = _exec_voice_action(best_sid)
            self.log_fn(f"✓ {msg}"); return

        # ── Inline open+play — "youtube pe despacito bajao" ──────
        ctx = self._detect_inline_context(t)
        if ctx:
            site, url = ctx
            self._browser_ctx = site
            webbrowser.open(url)
            self.log_fn(f"🌐 {site}: {url}")
            return

        # ── Local fallback FIRST — works without API key/credits ─────
        result = self._local_command_fallback(t)
        if result:
            self.log_fn(f"✓ {result}")
            return

        # ── Claude NLU fallback ───────────────────────────────────
        if not self._api_key or getattr(self, "_api_no_credits", False):
            self.log_fn(f"? Not understood: \"{t}\"  (no API credits)")
            return
        self.log_fn(f"🧠 Interpreting...")
        threading.Thread(target=self._nlu_interpret,
                         args=(t,), daemon=True).start()

    # ══════════════════════════════════════════════════════════════
    #  BROWSER CONTEXT ENGINE
    # ══════════════════════════════════════════════════════════════

    # Common play/search trigger words (Hindi + English)
    _PLAY_WORDS  = ["bajao","chala","chalao","laga","laga do","play karo",
                    "play kar","play","search karo","dhundo","suno","sun",
                    "dekho","chalu karo","chalu kar","shuru karo","on kar",
                    "dede","dikhao","lagao","start","run karo","bja","bja do"]
    _FILLER_WORDS = ["pe","par","mein","ko","ka","ki","ke","please","zara",
                     "bhai","yaar","jaldi","abhi","wala","wali","yeh","ye",
                     "voh","vo","mujhe","mujhe","ek","please","aur","toh"]

    def _detect_inline_context(self, text):
        """
        Detects: "youtube pe despacito bajao" type phrases.
        Returns (site_name, url) or None.
        """
        t = text.lower()
        # YouTube
        if any(w in t for w in ["youtube","yt"]):
            query = self._extract_query(t, ["youtube","yt"] + self._PLAY_WORDS + self._FILLER_WORDS)
            if query:
                url = f"https://www.youtube.com/results?search_query={query.replace(' ','+')}"
                return ("youtube", url)
            return ("youtube","https://www.youtube.com")
        # Google
        if any(w in t for w in ["google","chrome"]):
            query = self._extract_query(t, ["google","chrome"] + self._PLAY_WORDS + self._FILLER_WORDS)
            if query:
                url = f"https://www.google.com/search?q={query.replace(' ','+')}"
                return ("google", url)
        return None

    def _extract_query(self, text, remove_words):
        """Remove known filler/command words and return the remaining query."""
        q = text
        for w in sorted(remove_words, key=len, reverse=True):  # longest first
            q = q.replace(w, " ")
        return " ".join(q.split()).strip()

    def _browser_handle(self, text):
        """
        Handle voice when browser context is active.
        Returns True if handled.
        """
        t = text.strip().lower()
        ctx = self._browser_ctx

        # ── Exit context ──────────────────────────────────────────
        if any(e in t for e in ["context band","normal mode","reset context",
                                  "browser mode off","wapas normal"]):
            self.log_fn(f"🌐 {ctx} context off")
            self._browser_ctx = None; return True

        if ctx == "youtube":
            return self._yt_handle(t)
        if ctx in ("google","chrome"):
            return self._google_handle(t)
        return False

    def _yt_handle(self, text):
        """
        Smart YouTube handler.
        Priority:
          1. Number selection (if results list is showing)
          2. Playback controls
          3. Everything else = search query
        """
        t = text.strip().lower()

        # ── 1. Number selection from results list ─────────────────
        if hasattr(self, '_yt_results') and self._yt_results:
            num = self._parse_number(t)
            if num is not None:
                self._yt_play_number(num)
                return True
            # Also handle "pehla/first/top wala" etc.
            FIRST_WORDS = ["pehla","pehla wala","first","top","top wala",
                           "pehle wala","number one","ek number"]
            if any(w in t for w in FIRST_WORDS):
                self._yt_play_number(1); return True

        # ── Playback controls first ───────────────────────────────
        PAUSE_WORDS   = ["pause","rok","roko","ruk","ruk jao","band karo","thehro",
                         "roke","hold on","wait"]
        RESUME_WORDS  = ["resume","chala","chalao","dobara","phir chala","start karo",
                         "shuru karo","continue","unpause"]
        MUTE_WORDS    = ["mute","awaaz band","chup","silent","awaaz karo band",
                         "no sound","awaaz mat"]
        FULL_WORDS    = ["fullscreen","full screen","bada karo","full kar","poori screen",
                         "maximize","bada","bara karo"]
        FWD_WORDS     = ["aage","skip","fast forward","thoda aage","10 second aage",
                         "forward","age jao","aage skip"]
        BWD_WORDS     = ["peeche","rewind","thoda peeche","10 peeche","backward",
                         "peeche jao","peeche skip"]
        NEXT_WORDS    = ["next video","agla video","agla wala","next wala","agli video"]
        PREV_WORDS    = ["previous video","pichla video","pichla wala","previous wala"]
        HOME_WORDS    = ["youtube home","home page","home","youtube pe jao","main page",
                         "youtube kholo","kholo youtube","wapas youtube","youtube wapas"]
        VOLUP_WORDS   = ["louder","volume up","tej karo","tej kar","awaaz badao",
                         "awaz badao","volume badao","tej","zyada tej"]
        VOLDOWN_WORDS = ["quieter","volume down","dhima karo","dhima kar","awaaz kam",
                         "awaz kam","volume kam","kam karo","soft karo"]
        LIKE_WORDS    = ["like karo","like kar","like","thumbs up","pasand"]
        EXIT_WORDS    = ["youtube band","youtube close","youtube se bahar","context band",
                         "normal mode","reset","wapas normal"]

        # Close YouTube browser entirely
        CLOSE_YT_WORDS = ["youtube band karo","youtube band kro","youtube close karo",
                          "youtube close kro","youtube close","youtube band",
                          "youtube ko band karo","youtube ko close karo",
                          "browser band","browser close","browser ko band karo"]
        if any(w in t for w in CLOSE_YT_WORDS):
            self._browser_ctx = None
            self._close_browser()
            self.log_fn("📺 YouTube closed"); return True

        # Close current tab only
        CLOSE_TAB_WORDS = ["close tab","close current tab","tab band","tab band karo",
                           "tab close","is tab ko band karo","current tab band karo",
                           "yeh tab band","tab close karo","tab close kro"]
        if any(w in t for w in CLOSE_TAB_WORDS):
            self._close_current_tab()
            self.log_fn("📺 Tab closed"); return True

        if any(w in t for w in EXIT_WORDS):
            self._browser_ctx = None
            self.log_fn("📺 YouTube context off"); return True

        if any(w in t for w in PAUSE_WORDS):
            if PAG_OK: pyautogui.press("k")
            # Resume mic AND clear browser context
            # So next command (e.g. "open powerpoint") goes to normal NLU, not YT search
            def _pause_resume():
                time.sleep(0.5)
                self.resume_mic()
                self._browser_ctx = None   # exit YouTube context — listen for any command
                self.log_fn("🎤 Mic ON — say any command (YouTube context cleared)")
            threading.Thread(target=_pause_resume, daemon=True).start()
            self.log_fn("📺 ⏸ Video paused — mic resuming..."); return True

        if any(w in t for w in RESUME_WORDS) and not any(w in t for w in self._PLAY_WORDS):
            if PAG_OK: pyautogui.press("k")
            time.sleep(0.3)
            self.pause_mic()   # video playing again — mute mic
            self.log_fn("📺 ▶ Resume — mic paused"); return True

        if any(w in t for w in MUTE_WORDS):
            if PAG_OK: pyautogui.press("m")
            self.log_fn("📺 🔇 Mute"); return True

        if any(w in t for w in FULL_WORDS):
            if PAG_OK: pyautogui.press("f")
            self.log_fn("📺 ⛶ Fullscreen"); return True

        if any(w in t for w in FWD_WORDS):
            if PAG_OK: pyautogui.press("l")
            self.log_fn("📺 ⏩ +10s"); return True

        if any(w in t for w in BWD_WORDS):
            if PAG_OK: pyautogui.press("j")
            self.log_fn("📺 ⏪ -10s"); return True

        if any(w in t for w in NEXT_WORDS):
            if PAG_OK: pyautogui.hotkey("shift","n")
            self.log_fn("📺 ⏭ Next"); return True

        if any(w in t for w in PREV_WORDS):
            if PAG_OK: pyautogui.hotkey("shift","p")
            self.log_fn("📺 ⏮ Prev"); return True

        if any(w in t for w in LIKE_WORDS):
            if PAG_OK: pyautogui.press("shift+period")
            self.log_fn("📺 👍 Liked"); return True

        if any(w in t for w in VOLUP_WORDS):
            _exec_voice_action("volumeup"); self.log_fn("📺 🔊 Vol+"); return True

        if any(w in t for w in VOLDOWN_WORDS):
            _exec_voice_action("volumedown"); self.log_fn("📺 🔉 Vol-"); return True

        if any(w in t for w in HOME_WORDS):
            webbrowser.open("https://www.youtube.com"); return True

        # ── Scroll list / navigate results ────────────────────────
        # Works on search results page AND autocomplete suggestions
        DOWN_WORDS = ["niche","neeche","neche","down","go down","next",
                      "next result","aage","aage jao","neeche jao","niche jao",
                      "scroll down","neeche wala","next wala","skip",
                      "niche chalo","neechay"]
        UP_WORDS   = ["upar","oopar","up","go up","previous","peeche wala",
                      "upar jao","upar wala","upar chalo","scroll up",
                      "pichla","ek upar","peeche result"]
        OPEN_WORDS = ["open","enter","select","chalao",
                      "ye wala","is wala","open karo","open kar","play karo",
                      "play kar","is ko open karo","dekho","chala do","ye chala",
                      "is pe click karo","click","ok","haan",
                      "yes","choose","choose this","select this","ye lo"]

        if any(w == t or t.startswith(w) or (" "+w) in t or t.endswith(" "+w)
               for w in DOWN_WORDS):
            count = self._count_nav_steps(t, DOWN_WORDS)
            self._scroll_results("down", count)
            self.log_fn(f"📺 ⬇ {count} step{'s' if count>1 else ''}")
            return True

        if any(w == t or t.startswith(w) or (" "+w) in t or t.endswith(" "+w)
               for w in UP_WORDS):
            count = self._count_nav_steps(t, UP_WORDS)
            self._scroll_results("up", count)
            self.log_fn(f"📺 ⬆ {count} step{'s' if count>1 else ''}")
            return True

        if any(w == t or w in t for w in OPEN_WORDS):
            self._open_selected_result()
            self.log_fn("📺 ↩ Opening selected")
            return True

        # ── INTENT CLASSIFICATION before searching ────────────────
        # Don't blindly search everything — classify first.
        # Only search if intent is clearly "search for something".
        # Commands, gossip, selection phrases → don't search.
        intent = self._classify_yt_intent(t)

        if intent == "select":
            # User wants to select/play from current results
            num = self._parse_number(t)
            if num and hasattr(self, '_yt_results') and self._yt_results:
                self._yt_play_number(num); return True
            # Generic select without number — open focused item
            self._open_selected_result()
            self.log_fn("📺 ↩ Opening selected"); return True

        elif intent == "search":
            # Clearly a search query — extract and search
            STRIP_WORDS = (
                self._PLAY_WORDS + self._FILLER_WORDS +
                ["youtube","yt","search kar","search karo","dhundo",
                 "find karo","lagao","baja","baja do","bja","kro","krdo",
                 "kar do","kardo","chal","dekhna","sunna"]
            )
            query = t
            for w in sorted(STRIP_WORDS, key=len, reverse=True):
                query = (" " + query + " ").replace(" " + w + " ", " ").strip()
            query = " ".join(query.split()).strip()
            if query and len(query) >= 2:
                self._yt_search_in_tab(query)
                self.log_fn(f"📺 🔍 Searching: \"{query}\"")
                return True

        elif intent == "ignore":
            # Gossip / random talk / unclear — silently ignore
            self.log_fn(f"📺 Ignored: \"{t}\"")
            return True

        # intent == "unclear" and no API → ignore
        return True

        return False   # empty query — don't handle

    def _yt_search_in_tab(self, query):
        """Search YouTube using keyboard shortcuts in the open browser."""
        self.log_fn(f"📺 Searching: {query}")
        if not PAG_OK:
            webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ','+')}"); return
        try:
            time.sleep(0.2)
            pyautogui.press("/")        # YouTube shortcut: focus search bar
            time.sleep(0.4)
            pyautogui.hotkey("ctrl","a")
            time.sleep(0.1)
            pyautogui.press("delete")
            time.sleep(0.1)
            pyautogui.write(query, interval=0.04)
            time.sleep(0.2)
            pyautogui.press("enter")
            # Pause mic — browser may play audio (ads/previews on results page)
            # User can say "pause" or show palm to resume
            time.sleep(1.5)
            self.pause_mic()
        except Exception:
            webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ','+')}")
            time.sleep(1.5)
            self.pause_mic()



    def _fetch_yt_results(self, query):
        """
        Fetch YouTube search page HTML and extract video data.
        Uses ytInitialData JSON embedded in the page — most reliable.
        """
        import urllib.request, json as _json, re as _re
        results = []
        try:
            url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            })
            with urllib.request.urlopen(req, timeout=8) as r:
                html = r.read().decode("utf-8", errors="ignore")

            # Extract ytInitialData JSON
            m = _re.search(r"var ytInitialData\s*=\s*(\{.+?\});\s*</script>",
                           html, _re.DOTALL)
            if not m:
                m = _re.search(r"ytInitialData\s*=\s*(\{.+?\});", html, _re.DOTALL)
            if not m:
                self.log_fn("📺 Could not find ytInitialData"); return results

            data = _json.loads(m.group(1))

            # Walk the JSON tree for videoRenderer objects
            def walk(obj, depth=0):
                if not obj or depth > 10: return
                if isinstance(obj, dict):
                    if "videoRenderer" in obj:
                        v = obj["videoRenderer"]
                        try:
                            title = v["title"]["runs"][0]["text"]
                            vid   = v["videoId"]
                            url_v = f"https://www.youtube.com/watch?v={vid}"
                            ch    = v.get("ownerText",{}).get("runs",[{}])[0].get("text","")
                            dur   = v.get("lengthText",{}).get("simpleText","")
                            vws   = v.get("viewCountText",{}).get("simpleText","")
                            thumbs= v.get("thumbnail",{}).get("thumbnails",[])
                            thumb = thumbs[-1]["url"] if thumbs else                                     f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"
                            if thumb.startswith("//"): thumb = "https:"+thumb
                            results.append({
                                "title":title,"url":url_v,"channel":ch,
                                "duration":dur,"views":vws,"thumb":thumb
                            })
                        except Exception: pass
                        return
                    for v in obj.values(): walk(v, depth+1)
                elif isinstance(obj, list):
                    for item in obj[:30]: walk(item, depth+1)

            walk(data)
            self.log_fn(f"📺 Scraped {len(results)} results from YouTube")

        except Exception as e:
            self.log_fn(f"📺 Fetch error: {e}")
        return results[:12]


    def _classify_yt_intent(self, text):
        """
        Classify what the user wants in YouTube context.
        Returns: "search" | "select" | "ignore" | "unclear"

        Rules (fast local, no API needed for most cases):
          search  — explicit search trigger word present
          select  — selection/play trigger for current results
          ignore  — conversational filler / gossip
          unclear — ambiguous, needs Claude NLU
        """
        t = text.strip().lower()
        words = t.split()

        # ── SEARCH triggers — explicit intent to search ───────────
        SEARCH_TRIGGERS = [
            # English
            "search","find","look up","look for","show me","play",
            # Hindi/Hinglish — these clearly mean "search for X"
            "dhundo","search karo","dikhao","search kar","dhoondo",
            "bajao","chala","chalao","laga","lagao","suno","sun",
            "dekhna hai","sunna hai","find karo","khojo",
        ]
        if any(w in t for w in SEARCH_TRIGGERS):
            return "search"

        # ── SELECT triggers — pick from current list ──────────────
        SELECT_TRIGGERS = [
            # Numbers (already handled above but catch phrases here)
            "pehla","pehla wala","pehle wala","first wala","first one",
            "doosra","doosra wala","second wala","dusra",
            "teesra","teesra wala","third wala","tisra",
            "chautha","chautha wala","fourth wala",
            "paanchwa","fifth wala",
            # Generic select
            "start kro","start karo","shuru karo","shuru kar",
            "open karo","open kar","chalu karo","chalu kar",
            "is ko chala","ye chala","ye wala chala","yeh wala",
            "play karo","play kar","ye play karo",
            "isko","ise","is wale ko","usse","usko",
        ]
        if any(w in t for w in SELECT_TRIGGERS):
            return "select"

        # ── IGNORE — clearly conversational / gossip ──────────────
        GOSSIP_PATTERNS = [
            # Common conversation fillers
            "kya kar rahe","kya hua","theek hai","accha","haan","nahi",
            "ok ok","okay","hmm","arre","yaar suno","bhai suno",
            "matlab","samjhe","samjha","pata hai","dekho bhai",
            "ek second","ruko","thoda ruko","sun","suno yaar",
            "kya baat","wah","bahut accha","mast hai","mast",
            "acha hai","yeh kya","kya yeh","yeh dekho",
        ]
        if any(p in t for p in GOSSIP_PATTERNS):
            return "ignore"

        # ── Short utterances likely to be noise ───────────────────
        KEEP_SINGLES = {"pause","mute","next","skip","back","home",
                        "youtube","play","stop","forward","rewind",
                        "fullscreen","like","upar","niche","open"}
        if len(words) <= 1 and t not in KEEP_SINGLES:
            return "ignore"

        # ── Medium phrases — could go either way ─────────────────
        # If results are showing, lean toward "select" for short phrases
        if hasattr(self, '_yt_results') and self._yt_results:
            num = self._parse_number(t)
            if num: return "select"
            # Short phrase with no clear intent → probably select, not search
            if len(words) <= 3:
                return "select"

        # ── Use Claude if API key set ─────────────────────────────
        if self._api_key:
            return self._claude_classify_yt(t)

        # No API — be conservative, don't search ambiguous things
        return "ignore"

    def _claude_classify_yt(self, text):
        """Ask Claude to classify intent in YouTube context."""
        try:
            import urllib.request, json as _json
            has_results = hasattr(self, '_yt_results') and bool(self._yt_results)
            ctx = "Search results are currently showing on screen." if has_results \
                  else "No search results are currently showing."

            system = f"""You are classifying voice commands in a YouTube controller.
{ctx}

The user speaks Hindi, English, or Hinglish.
Reply with ONLY one word: search | select | ignore

search  = user wants to search for a video (e.g. "arijit singh ke gaane", "javascript tutorial")
select  = user wants to play/open something from current results (e.g. "first wala start kro", "doosra wala", "chautha chala")
ignore  = random talk, gossip, filler, unclear (e.g. "theek hai yaar", "haan", "kya hua")

Examples:
"first wala start kro"     → select
"javascript course dikhao" → search
"arijit singh bajao"       → search
"doosra wala open karo"    → select
"theek hai yaar"           → ignore
"haan"                     → ignore
"char number wala"         → select
"lofi music chala"         → search
"ok ok"                    → ignore"""

            payload = _json.dumps({
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "system": system,
                "messages": [{"role":"user","content": text}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={"Content-Type":"application/json",
                         "anthropic-version":"2023-06-01",
                         "x-api-key":self._api_key},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = _json.loads(resp.read())
            result = data["content"][0]["text"].strip().lower()
            self.log_fn(f"🧠 intent → {result}")
            return result if result in ("search","select","ignore") else "ignore"
        except Exception as e:
            err = str(e)
            if "401" in err or "400" in err:
                self.log_fn("🧠 ⚠ API key invalid/expired")
            return "ignore"   # on error, don't accidentally search

    def _yt_play_number(self, num):
        """Open result #num in the default browser."""
        if not hasattr(self, '_yt_results') or not self._yt_results:
            self.log_fn("⚠ No results — search something first"); return
        if num < 1 or num > len(self._yt_results):
            self.log_fn(f"⚠ No result #{num} (1–{len(self._yt_results)})"); return

        item  = self._yt_results[num - 1]
        title = item.get("title","") if isinstance(item,dict) else item[0]
        url   = item.get("url","")   if isinstance(item,dict) else item[1]
        is_yt = "youtube.com" in url or "youtu.be" in url

        self.log_fn(f"{'📺' if is_yt else '🔍'} Opening #{num}: {title[:55]}")

        # Open URL in the user's default browser
        webbrowser.open(url)
        if is_yt:
            time.sleep(1.0)
            self.pause_mic()

    def _open_selected_result(self):
        """Press Enter on focused element — opens selected result."""
        if PAG_OK:
            try: pyautogui.press("enter")
            except Exception: pass
        self.pause_mic()

    def _scroll_results(self, direction, count=1):
        """Scroll page up/down using keyboard arrow keys."""
        if not PAG_OK: return
        key = "down" if direction == "down" else "up"
        try:
            for _ in range(count):
                pyautogui.press(key)
                import time; time.sleep(0.1)
        except Exception: pass

    def _count_nav_steps(self, text, keywords):
        """Count steps from spoken text — 'niche niche' = 2, 'teen niche' = 3."""
        count = 0
        for kw in keywords[:4]:
            count += text.count(kw)
        num_words = {"ek":1,"do":2,"teen":3,"char":4,"paanch":5,
                     "one":1,"two":2,"three":3,"four":4,"five":5,
                     "2":2,"3":3,"4":4,"5":5}
        for word, n in num_words.items():
            if word in text.split():
                count = max(count, n); break
        return max(1, min(count, 10))

    def _browser_search_in_tab(self, query, engine="google"):
        """Search using requests scrape + open URL in default browser."""
        if engine == "youtube":
            url = f"https://www.youtube.com/results?search_query={query.replace(' ','+')}"
        else:
            url = f"https://www.google.com/search?q={query.replace(' ','+')}"
        webbrowser.open(url)

    def _close_browser(self):
        """Close browser tab via Ctrl+W (works on any browser)."""
        self._browser_ctx = None
        self._yt_results  = []
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","w")
            except Exception: pass
        self.log_fn("🌐 Tab closed")

    def _close_current_tab(self):
        """Close current browser tab via Ctrl+W."""
        if PAG_OK:
            try: pyautogui.hotkey("ctrl","w")
            except Exception: pass
        self.log_fn("🗂 Tab closed")


    def _google_handle(self, text):
        """
        Smart Google/Chrome handler.
        Any phrase = Google search after stripping known words.
        """
        t = text.strip().lower()

        EXIT_WORDS = ["google band","chrome band","context band","normal mode",
                      "reset","wapas normal"]
        if any(w in t for w in EXIT_WORDS):
            self._browser_ctx = None
            self.log_fn("🔍 Google context off"); return True

        # Strip known non-query words — rest is the search query
        STRIP = (["search","dhundo","karo","batao","find","google","chrome",
                  "pe search karo","se search karo","search kar"] +
                 self._FILLER_WORDS)
        query = t
        for w in sorted(STRIP, key=len, reverse=True):
            query = (" " + query + " ").replace(" " + w + " ", " ").strip()
        query = " ".join(query.split()).strip()

        if query and len(query) >= 2:
            self._browser_search_in_tab(query, "google")
            self.log_fn(f"🔍 Google: \"{query}\""); return True

        return False

    def _detect_location(self, text):
        """Check if text mentions a known location. Returns path or None."""
        for phrase, path in self.ROOT_LOCATIONS.items():
            if phrase in text:
                return path
        # "jake X kholo" / "X pe jao" / "open X" patterns
        open_words = ["jake","kholo","khol","open","jao","chalao","pe jao"]
        for phrase, path in self.ROOT_LOCATIONS.items():
            for ow in open_words:
                if phrase in text and ow in text:
                    return path
                # e.g. "local disk d ko kholo"
                if phrase in text:
                    return path
        return None

    def _nav_enter(self, path):
        """Enter a folder — open it in Explorer and keep tracking for next command."""
        # Shell paths (My Computer)
        if path.startswith("::"):
            subprocess.Popen(["explorer.exe", path])
            self.log_fn(f"📁 Opened: My Computer")
            # Can't navigate inside shell paths — stay at root
            self._nav_path    = "::"
            self._nav_items   = []
            return

        if not os.path.exists(path):
            self.log_fn(f"⚠ Not found: {path}"); return

        # Push history
        if self._nav_path and self._nav_path != "::":
            self._nav_history.append(self._nav_path)

        self._nav_path = path

        # Open in Explorer
        subprocess.Popen(["explorer.exe", path])
        folder_name = os.path.basename(path) or path
        self.log_fn(f"📁 Opened: {folder_name}  — say a folder/file name to continue")

        # Index contents silently (for name matching on next command)
        try:
            items = []
            for e in sorted(os.scandir(path),
                            key=lambda e: (not e.is_dir(), e.name.lower())):
                if not e.name.startswith('.'):
                    items.append((e.name, e.path, e.is_dir()))
            self._nav_items = items
        except Exception:
            self._nav_items = []

    def _nav_input(self, text):
        """
        Fully dynamic nav handler — understands any language/phrasing.
        Fast local matching first, Claude NLU fallback for anything unclear.
        """
        t = text.strip().lower()

        # ── Resolve intent (fast local first, NLU fallback) ──────
        intent = self._resolve_nav_intent(t)

        if intent == "down":
            # Count how many "down" signals in the phrase
            count = self._count_steps(t, ["niche","neeche","down","next","aage"])
            self._explorer_key("down", count)
            self.log_fn(f"⬇ {count} step{'s' if count>1 else ''}")

        elif intent == "up":
            count = self._count_steps(t, ["upar","up","previous","peeche","pichle"])
            self._explorer_key("up", count)
            self.log_fn(f"⬆ {count} step{'s' if count>1 else ''}")

        elif intent == "open":
            self._explorer_key("enter", 1)
            self.log_fn("📂 Opened selected item")

        elif intent == "back":
            if self._nav_history:
                prev = self._nav_history.pop()
                self._nav_path = None
                self._nav_enter(prev)
            else:
                self.log_fn("📁 Already at top")
                self._nav_reset()

        elif intent == "cancel":
            self.log_fn("📁 Navigation ended")
            self._nav_reset()

        elif intent == "navigate":
            # User said a new location — go there
            new_loc = self._detect_location(t)
            if new_loc:
                self._nav_history = []
                self._nav_path = None
                self._nav_enter(new_loc)
            else:
                self.log_fn(f"⚠ Location not understood: \"{t}\"")

        elif intent == "find":
            # User said a folder/file name — find and open it
            if self._nav_path and self._nav_path != "::" and self._nav_items:
                match = self._best_match(t)
                if match:
                    name, fpath, is_dir = match
                    if is_dir:
                        self.log_fn(f"📁 → {name}")
                        self._nav_enter(fpath)
                    else:
                        self._open_file(fpath)
                        self.log_fn(f"📄 Opened: {name}")
                        self._nav_reset()
                else:
                    self.log_fn(f"🔎 Searching \"{t}\"...")
                    threading.Thread(target=self._deep_find,
                                     args=(t,), daemon=True).start()
            else:
                self.log_fn(f"⚠ Not in a folder yet")

        else:
            # intent == "unknown" — if API key, try NLU
            if self._api_key:
                threading.Thread(target=self._nlu_nav,
                                 args=(t,), daemon=True).start()
            else:
                # No API — try best-effort local
                new_loc = self._detect_location(t)
                if new_loc:
                    self._nav_history = []; self._nav_path = None
                    self._nav_enter(new_loc); return
                if self._nav_items:
                    match = self._best_match(t)
                    if match:
                        name, fpath, is_dir = match
                        if is_dir: self._nav_enter(fpath)
                        else: self._open_file(fpath); self._nav_reset()
                        return
                self.log_fn(f"? Not understood: \"{t}\"")

    def _resolve_nav_intent(self, t):
        """
        Fast local resolution — covers common Hindi/English/Hinglish patterns.
        Returns: down | up | open | back | cancel | navigate | find | unknown
        """
        # ── DOWN ─────────────────────────────────────────────────
        DOWN = ["niche","neeche","neche","nichey","down","go down","move down",
                "next","next item","aage","aage jao","aage badho",
                "next wala","neeche jao","neeche chalo","scroll down",
                "arrow down","move forward","skip","skip karo"]
        if any(w == t or t.startswith(w+" ") or (" "+w) in t or t.endswith(" "+w)
               for w in DOWN):
            return "down"

        # ── UP ───────────────────────────────────────────────────
        UP = ["upar","oopar","up","go up","move up","upar jao","upar chalo",
              "previous","pichla","pichle","peeche wala","back item",
              "arrow up","ek upar","ek upar jao","scroll up"]
        if any(w == t or t.startswith(w+" ") or (" "+w) in t or t.endswith(" "+w)
               for w in UP):
            return "up"

        # ── OPEN ─────────────────────────────────────────────────
        OPEN = ["open","kholo","khol do","khol","open karo","open kar",
                "enter","select","select karo","is ko kholo","ye kholo",
                "open it","open this","isko open karo","open kar do",
                "andar jao","is me jao","chalao","run","chala","launch"]
        if any(w == t or t.startswith(w+" ") or (" "+w) in t or t.endswith(" "+w)
               for w in OPEN):
            return "open"

        # ── BACK ─────────────────────────────────────────────────
        BACK = ["back","go back","wapas","wapas jao","peeche","peeche jao",
                "peeche chalo","ek peeche","ek step back","previous folder",
                "parent folder","bahar jao","exit folder","folder se bahar",
                "backspace","back karo","wapas aa jao","nikal"]
        if any(w == t or t.startswith(w+" ") or (" "+w) in t or t.endswith(" "+w)
               for w in BACK):
            return "back"

        # ── CANCEL ───────────────────────────────────────────────
        CANCEL = ["cancel","stop","exit","quit","close","band karo","bas",
                  "done","ho gaya","nikal jao","band","finish","khatam",
                  "navigation band karo","close navigation","bnd kro"]
        if any(w == t for w in CANCEL):
            return "cancel"

        # ── NAVIGATE to a known location ─────────────────────────
        if self._detect_location(t):
            return "navigate"

        # ── FIND by name ─────────────────────────────────────────
        # If text looks like a name (not a command), treat as find
        # Heuristic: if it has >2 chars and no command words → find
        command_words = set(w for lst in [DOWN,UP,OPEN,BACK,CANCEL] for w in lst)
        if len(t) > 2 and t not in command_words:
            return "find"

        return "unknown"

    def _count_steps(self, text, keywords):
        """Count how many movement steps from spoken text."""
        # "niche niche" = 2, "niche niche niche" = 3
        count = 0
        for kw in keywords:
            count += text.count(kw)
        # Also handle number words: "teen niche" = 3 steps
        num_words = {"ek":1,"do":2,"teen":3,"char":4,"paanch":5,
                     "one":1,"two":2,"three":3,"four":4,"five":5,
                     "2":2,"3":3,"4":4,"5":5}
        for word, n in num_words.items():
            if word in text.split():
                count = max(count, n); break
        return max(1, min(count, 10))

    def _nlu_nav(self, text):
        """
        Ask Claude what the user means in navigation context.
        Returns nav intent and executes it.
        """
        try:
            import urllib.request, json as _json
            folder = os.path.basename(self._nav_path or "") or "root"
            system = f"""You are inside a file explorer navigation session.
Current folder: {folder}
User speaks Hindi, English, or Hinglish.
Reply with ONLY one word from: down, up, open, back, cancel, find

MEANING:
down  = move selection down / next item / niche / aage
up    = move selection up / previous item / upar / peeche wala
open  = open/enter selected item / kholo / enter / chalao
back  = go to parent folder / wapas / peeche / exit folder
cancel= end navigation / band karo / done / bas
find  = user said a folder or file name to search for"""

            payload = _json.dumps({
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "system": system,
                "messages": [{"role":"user","content": text}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={"Content-Type":"application/json",
                         "anthropic-version":"2023-06-01",
                         "x-api-key":self._api_key},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = _json.loads(resp.read())
            intent = data["content"][0]["text"].strip().lower()
            self.log_fn(f"🧠 nav → {intent}")
            # Re-run with resolved intent
            self._execute_nav_intent(intent, text)
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err or "400" in err:
                self.log_fn("🧠 ⚠ API key issue — check console.anthropic.com")
            else:
                self.log_fn(f"🧠 NLU error: {e}")

    def _execute_nav_intent(self, intent, original_text):
        """Execute a resolved nav intent string."""
        if intent == "down":
            self._explorer_key("down", 1); self.log_fn("⬇")
        elif intent == "up":
            self._explorer_key("up", 1); self.log_fn("⬆")
        elif intent == "open":
            self._explorer_key("enter", 1); self.log_fn("📂 Opened")
        elif intent == "back":
            if self._nav_history:
                prev = self._nav_history.pop()
                self._nav_path = None; self._nav_enter(prev)
            else:
                self._nav_reset()
        elif intent == "cancel":
            self.log_fn("📁 Navigation ended"); self._nav_reset()
        elif intent == "find":
            if self._nav_items:
                match = self._best_match(original_text)
                if match:
                    name, fpath, is_dir = match
                    if is_dir: self._nav_enter(fpath)
                    else: self._open_file(fpath); self._nav_reset()
                else:
                    threading.Thread(target=self._deep_find,
                                     args=(original_text,), daemon=True).start()

    def _explorer_key(self, key, count=1):
        """Send keypress to foreground window (Explorer). Reliable via ctypes."""
        KEY_MAP = {"down":0x28,"up":0x26,"enter":0x0D,
                   "backspace":0x08,"left":0x25,"right":0x27}
        vk = KEY_MAP.get(key)
        if vk is None: return
        try:
            import ctypes as _ct
            for _ in range(count):
                _ct.windll.user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.04)
                _ct.windll.user32.keybd_event(vk, 0, 0x0002, 0)
                time.sleep(0.05)
        except Exception:
            if PAG_OK:
                try:
                    for _ in range(count): pyautogui.press(key)
                except Exception: pass




    def _best_match(self, query):
        """
        Find the single best matching item in current folder.
        Priority: exact > startswith > all-words > any-word.
        Returns (name, path, is_dir) or None.
        """
        q       = query.lower().strip()
        q_words = [w for w in q.split() if len(w) > 1]
        best    = None
        best_score = 0

        for name, fpath, is_dir in self._nav_items:
            n       = name.lower()
            n_clean = os.path.splitext(n)[0]   # without extension
            score   = 0

            if n_clean == q or n == q:
                return (name, fpath, is_dir)   # exact — return immediately

            if n_clean.startswith(q) or n.startswith(q):
                score = 90
            elif all(w in n for w in q_words):
                score = 70
            elif sum(1 for w in q_words if w in n) >= max(1, len(q_words)-1):
                score = 50
            elif any(w in n for w in q_words):
                score = 30

            if score > best_score:
                best_score = score
                best       = (name, fpath, is_dir)

        return best if best_score >= 30 else None

    def _deep_find(self, query):
        """Search recursively inside current folder — open best match directly."""
        q_words = [w for w in query.lower().split() if len(w) > 1]
        found   = []

        try:
            for dirpath, dirnames, filenames in os.walk(self._nav_path):
                depth = dirpath.replace(self._nav_path, "").count(os.sep)
                if depth > 5: dirnames.clear(); continue
                dirnames[:] = [d for d in dirnames
                               if not d.startswith('.') and
                               d not in ("__pycache__",".git","node_modules",
                                         "$RECYCLE.BIN","System Volume Information")]
                for entry in dirnames + filenames:
                    el = entry.lower()
                    if all(w in el for w in q_words):
                        full = os.path.join(dirpath, entry)
                        found.append(full)
                        if len(found) >= 5: break
                if len(found) >= 5: break
        except Exception: pass

        if not found:
            self.log_fn(f"⚠ \"{query}\" not found")
            return

        # Open best match directly (prefer folders, then files)
        folders = [p for p in found if os.path.isdir(p)]
        target  = folders[0] if folders else found[0]

        if os.path.isdir(target):
            self.log_fn(f"📁 Found → {os.path.basename(target)}")
            self._nav_enter(target)
        else:
            self._open_file(target)
            self.log_fn(f"📄 Opened: {os.path.basename(target)}")
            self._nav_reset()


    def _open_file(self, path):
        try:
            os.startfile(path)
        except AttributeError:
            subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.log_fn(f"⚠ Cannot open: {e}")

    def _nav_reset(self):
        self._nav_path    = None
        self._nav_items   = []
        self._nav_history = []

    def _parse_number(self, text):
        """
        Spoken number → int. Scans full text so
        'doosra wala start kro' → 2, 'teen number chala' → 3.
        """
        NUMS = {
            "one":1,"two":2,"three":3,"four":4,"five":5,
            "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
            "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
            "ek":1,"do":2,"teen":3,"char":4,"paanch":5,
            "chhe":6,"saat":7,"aath":8,"nau":9,"das":10,
            "first":1,"second":2,"third":3,"fourth":4,"fifth":5,
            "pehla":1,"pehli":1,"pehle":1,"pehla wala":1,"pehle wala":1,
            "doosra":2,"dusra":2,"doosri":2,"doosre":2,"doosra wala":2,
            "teesra":3,"tisra":3,"teesri":3,"teesre":3,"teesra wala":3,
            "chautha":4,"chauthi":4,"chauthe":4,"chautha wala":4,
            "paanchwa":5,"paanchvi":5,"paanchwe":5,"paanchwa wala":5,
            "1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,
        }
        t = text.strip().lower()
        # Multi-word keys first (longest wins)
        for key in sorted(NUMS, key=len, reverse=True):
            if key in t:
                return NUMS[key]
        # Bare digit token
        for tok in t.split():
            if tok.isdigit():
                return int(tok)
        return None

    # ══════════════════════════════════════════════════════════════
    #  CLAUDE NLU FALLBACK
    # ══════════════════════════════════════════════════════════════
    def _local_command_fallback(self, text):
        """
        Fast local pattern match for common commands — works WITHOUT API key.
        Handles open_app, system commands in English + Hindi/Hinglish.
        Returns status string if handled, None otherwise.
        """
        t = text.lower().strip()

        # App open patterns — "X kholo / X open karo / open X / X chalu karo"
        APP_PATTERNS = {
            # Office
            "powerpoint":  ["start","powerpnt"],
            "power point": ["start","powerpnt"],
            "ppt":         ["start","powerpnt"],
            "har point":   ["start","powerpnt"],   # Hinglish mishear
            "presentation":["start","powerpnt"],
            "excel":       ["start","excel"],
            "word":        ["start","winword"],
            "ms word":     ["start","winword"],
            "access":      ["start","msaccess"],
            "outlook":     ["start","outlook"],
            "onenote":     ["start","onenote"],
            # Browsers
            "chrome":      ["start","chrome"],
            "google chrome":["start","chrome"],
            "firefox":     ["start","firefox"],
            "edge":        ["start","msedge"],
            # Media
            "vlc":         ["start","vlc"],
            "spotify":     ["start","spotify"],
            "music":       ["start","spotify"],
            "netflix":     ["start","netflix"],
            # Tools
            "notepad":     ["notepad.exe"],
            "calculator":  ["calc.exe"],
            "calc":        ["calc.exe"],
            "paint":       ["mspaint.exe"],
            "cmd":         ["cmd.exe"],
            "terminal":    ["cmd.exe"],
            "file explorer":["explorer.exe"],
            "explorer":    ["explorer.exe"],
            # Communication
            "whatsapp":    ["start","whatsapp"],
            "telegram":    ["start","telegram"],
            "teams":       ["start","teams"],
            "zoom":        ["start","zoom"],
            "discord":     ["start","discord"],
            "skype":       ["start","skype"],
            # Dev
            "vscode":      ["start","code"],
            "vs code":     ["start","code"],
            "visual studio":["start","code"],
            "pycharm":     ["start","pycharm"],
            "android studio":["start","studio64"],
            # System
            "photoshop":   ["start","photoshop"],
            "settings":    ["start","ms-settings:"],
            "task manager":["taskmgr.exe"],
            "taskmanager": ["taskmgr.exe"],
        }

        # Clean text — remove filler words for matching
        FILLERS = ["kholo","karo","open","chalu","karna","start","bhai",
                   "yaar","please","zara","do","kar","dedo","de do",
                   "chalao","launch","run","mujhe","mera","meri"]
        t_clean = t
        for f in sorted(FILLERS, key=len, reverse=True):
            t_clean = t_clean.replace(f, " ")
        t_clean = " ".join(t_clean.split()).strip()

        # Check app name in original OR cleaned text
        for app, cmd in APP_PATTERNS.items():
            if app in t or app in t_clean:
                try:
                    import subprocess
                    subprocess.Popen(cmd, shell=(cmd[0]=="start"))
                    return f"Opening {app}"
                except Exception:
                    # Fallback: Windows search
                    if PAG_OK:
                        try:
                            import time as _t
                            pyautogui.hotkey("win")
                            _t.sleep(0.5)
                            pyautogui.write(app, interval=0.05)
                            _t.sleep(0.8)
                            pyautogui.press("enter")
                            return f"Opening {app} via search"
                        except Exception: pass

        # System command patterns
        SYSTEM_LOCAL = {
            ("volume up","volume badhao","louder","tej karo","awaaz badao"):
                lambda: [pyautogui.press("volumeup") for _ in range(3)] if PAG_OK else None,
            ("volume down","volume kam","quieter","dhima karo","awaaz kam"):
                lambda: [pyautogui.press("volumedown") for _ in range(3)] if PAG_OK else None,
            ("mute","awaaz band","silent"):
                lambda: pyautogui.press("volumemute") if PAG_OK else None,
            ("screenshot","screenshot lo"):
                lambda: pyautogui.hotkey("win","shift","s") if PAG_OK else None,
            ("minimize","chota karo"):
                lambda: pyautogui.hotkey("win","down") if PAG_OK else None,
            ("copy","copy karo"):
                lambda: pyautogui.hotkey("ctrl","c") if PAG_OK else None,
            ("paste","paste karo"):
                lambda: pyautogui.hotkey("ctrl","v") if PAG_OK else None,
            ("undo","undo karo","wapas"):
                lambda: pyautogui.hotkey("ctrl","z") if PAG_OK else None,
            ("save","save karo"):
                lambda: pyautogui.hotkey("ctrl","s") if PAG_OK else None,
        }
        for triggers, action in SYSTEM_LOCAL.items():
            if any(tr in t for tr in triggers):
                try:
                    action()
                    return f"Done: {t[:30]}"
                except Exception: pass

        return None  # not handled locally

    def _nlu_interpret(self, text):
        """
        Send any voice command to Claude API.
        Claude returns JSON with action + parameters.
        Handles: open apps, websites, search, system commands,
                 Hinglish, casual speech, typos — everything.
        """
        if not self._api_key:
            # Try local fallback first
            result = self._local_command_fallback(text)
            if result:
                self.log_fn(f"✓ {result}")
                return
            self.log_fn(f"? Unknown: \"{text}\"")
            return

        try:
            import urllib.request, json as _json

            system_prompt = """You are a voice command interpreter for a Windows PC controller.
The user speaks English, Hindi, or Hinglish. They may have just paused a YouTube video.

CRITICAL RULE: "open X" / "X kholo" / "X chalu karo" = open_app or open_url — NEVER search.
App names like powerpoint, excel, word, notepad, spotify, chrome, vlc are ALWAYS open_app.

Return ONLY valid JSON — no markdown, no explanation:
{"action": "<action>", "param": "<param>"}

ACTIONS:
- open_app  → open a desktop application
              param = exact app name: "powerpoint", "excel", "word", "notepad", "chrome",
                      "spotify", "vlc", "whatsapp", "telegram", "calculator", "paint", "cmd"
- open_url  → open a website in browser
              param = URL e.g. "https://facebook.com"
- youtube   → open YouTube website
              param = ""
- system    → PC system command
              param = screenshot | lock | taskmanager | shutdown | restart |
                      volumeup | volumedown | mute | minimize | maximize | close |
                      copy | paste | cut | undo | save | selectall |
                      zoomin | zoomout | playpause | nexttrack | prevtrack |
                      showdesktop | goback | goforward
- search    → explicitly search Google (user said "search", "google", "dhundo")
              param = search query
              ONLY use when user clearly says search/google/dhundo etc.
              Do NOT use for app names or task commands.
- none      → casual talk, filler, unclear, not a command
              param = ""

DECISION RULES:
1. Any app name → open_app (even if phrased casually)
2. "X kholo / X chalu karo / X open karo / X start karo" → open_app or open_url
3. Only use "search" if user explicitly says search/google/dhundo/find
4. System words → system action
5. Vague or conversational → none

EXAMPLES (memorize these patterns):
"powerpoint kholo"         → {"action":"open_app","param":"powerpoint"}
"open powerpoint"          → {"action":"open_app","param":"powerpoint"}
"powerpoint chalu karo"    → {"action":"open_app","param":"powerpoint"}
"excel mein kaam karna hai"→ {"action":"open_app","param":"excel"}
"word kholo bhai"          → {"action":"open_app","param":"word"}
"spotify pe music"         → {"action":"open_app","param":"spotify"}
"chrome kholo"             → {"action":"open_app","param":"chrome"}
"volume badhao"            → {"action":"system","param":"volumeup"}
"awaaz band karo"          → {"action":"system","param":"mute"}
"screenshot lo"            → {"action":"system","param":"screenshot"}
"google pe python dhundo"  → {"action":"search","param":"python"}
"python tutorial dhundo"   → {"action":"search","param":"python tutorial"}
"facebook.com"             → {"action":"open_url","param":"https://facebook.com"}
"youtube kholo"            → {"action":"youtube","param":""}
"theek hai yaar"           → {"action":"none","param":""}
"haan"                     → {"action":"none","param":""}
"kya hua"                  → {"action":"none","param":""}
"""

            payload = _json.dumps({
                "model":      "claude-sonnet-4-5",
                "max_tokens": 150,
                "system":     system_prompt,
                "messages":   [{"role":"user","content": text}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={"Content-Type":       "application/json",
                         "anthropic-version":  "2023-06-01",
                         "x-api-key":          self._api_key},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = _json.loads(resp.read())

            raw = data["content"][0]["text"].strip()
            # Strip markdown fences if any
            raw = raw.replace("```json","").replace("```","").strip()
            result = _json.loads(raw)

            action = result.get("action","none").strip().lower()
            param  = result.get("param","").strip()

            self.log_fn(f"🧠 {action}: {param!r}")

            # ── Execute the action ────────────────────────────────
            if action == "none" or not action:
                return

            if action == "youtube":
                self._browser_ctx = "youtube"
                _exec_voice_action("youtube")
                self.log_fn("📺 YouTube opened")
                return

            if action == "open_url":
                url = param if param.startswith("http") else f"https://{param}"
                webbrowser.open(url)
                self.log_fn(f"🌐 {url}")
                return

            if action == "search":
                webbrowser.open(f"https://www.google.com/search?q={param.replace(' ','+')}")
                self.log_fn(f"🔍 {param}")
                return

            if action == "open_app":
                self._open_app(param)
                return

            if action == "system":
                self._exec_system(param)
                return

            if action == "navigate":
                path = self._detect_location(param or text)
                if path: self._nav_enter(path)
                return

        except Exception as e:
            err = str(e)
            if "401" in err:
                self.log_fn("🧠 ⚠ API key invalid or expired — get a new key at console.anthropic.com")
            elif "403" in err:
                self.log_fn("🧠 ⚠ API key has no access — check your Anthropic account")
            elif "429" in err:
                self.log_fn("🧠 ⚠ API rate limit hit — wait a moment")
            elif "400" in err:
                # Read actual error body
                try:
                    body = e.read().decode() if hasattr(e,'read') else err
                    self.log_fn(f"🧠 ⚠ API 400 error: {body[:120]}")
                    if "credit balance" in body or "too low" in body:
                        self._api_no_credits = True
                        self.log_fn("💳 API credits exhausted — running in local-only mode")
                        self.log_fn("   → Recharge at console.anthropic.com or use local commands")
                except Exception:
                    self.log_fn(f"🧠 ⚠ API key may be expired — check console.anthropic.com")
            else:
                self.log_fn(f"🧠 NLU error: {e}")

    def _open_app(self, app_name):
        """Open any app by name — tries multiple methods."""
        import subprocess
        a = app_name.lower().strip()
        self.log_fn(f"🚀 Opening: {a}")

        # Direct mappings
        APP_MAP = {
            "chrome":      ["start","chrome"],
            "firefox":     ["start","firefox"],
            "edge":        ["start","msedge"],
            "notepad":     ["notepad.exe"],
            "word":        ["start","winword"],
            "excel":       ["start","excel"],
            "powerpoint":  ["start","powerpnt"],
            "vlc":         ["start","vlc"],
            "spotify":     ["start","spotify"],
            "whatsapp":    ["start","whatsapp"],
            "telegram":    ["start","telegram"],
            "calculator":  ["calc.exe"],
            "paint":       ["mspaint.exe"],
            "explorer":    ["explorer.exe"],
            "cmd":         ["cmd.exe"],
            "task manager":["taskmgr.exe"],
            "taskmanager": ["taskmgr.exe"],
            "settings":    ["start","ms-settings:"],
            "camera":      ["start","microsoft.windows.camera:"],
            "photos":      ["start","ms-photos:"],
            "mail":        ["start","outlookmail:"],
        }
        # Check direct map
        for key, cmd in APP_MAP.items():
            if key in a or a in key:
                try:
                    subprocess.Popen(cmd, shell=(cmd[0]=="start"))
                    return
                except Exception: pass

        # Try shell "start <name>" — works for many installed apps
        try:
            subprocess.Popen(["start", a], shell=True)
            return
        except Exception: pass

        # Try searching Windows
        if PAG_OK:
            try:
                pyautogui.hotkey("win")
                import time; time.sleep(0.5)
                pyautogui.write(a, interval=0.05)
                time.sleep(0.8)
                pyautogui.press("enter")
                return
            except Exception: pass

        self.log_fn(f"⚠ Could not open: {a}")

    def _exec_system(self, cmd):
        """Execute a system command by name."""
        import subprocess
        c = cmd.lower().strip()
        self.log_fn(f"⚙ System: {c}")

        CMD_MAP = {
            "screenshot":  lambda: pyautogui.hotkey("win","shift","s") if PAG_OK else None,
            "lock":        lambda: ctypes.windll.user32.LockWorkStation() if CTYPES_OK else None,
            "taskmanager": lambda: pyautogui.hotkey("ctrl","shift","esc") if PAG_OK else None,
            "shutdown":    lambda: subprocess.Popen(["shutdown","/s","/t","0"]),
            "restart":     lambda: subprocess.Popen(["shutdown","/r","/t","0"]),
            "volumeup":    lambda: [pyautogui.press("volumeup") for _ in range(3)] if PAG_OK else None,
            "volumedown":  lambda: [pyautogui.press("volumedown") for _ in range(3)] if PAG_OK else None,
            "mute":        lambda: pyautogui.press("volumemute") if PAG_OK else None,
            "minimize":    lambda: pyautogui.hotkey("win","down") if PAG_OK else None,
            "maximize":    lambda: pyautogui.hotkey("win","up") if PAG_OK else None,
            "close":       lambda: pyautogui.hotkey("alt","f4") if PAG_OK else None,
            "copy":        lambda: pyautogui.hotkey("ctrl","c") if PAG_OK else None,
            "paste":       lambda: pyautogui.hotkey("ctrl","v") if PAG_OK else None,
            "cut":         lambda: pyautogui.hotkey("ctrl","x") if PAG_OK else None,
            "undo":        lambda: pyautogui.hotkey("ctrl","z") if PAG_OK else None,
            "save":        lambda: pyautogui.hotkey("ctrl","s") if PAG_OK else None,
            "selectall":   lambda: pyautogui.hotkey("ctrl","a") if PAG_OK else None,
            "zoomin":      lambda: pyautogui.hotkey("ctrl","=") if PAG_OK else None,
            "zoomout":     lambda: pyautogui.hotkey("ctrl","-") if PAG_OK else None,
            "playpause":   lambda: pyautogui.press("playpause") if PAG_OK else None,
            "nexttrack":   lambda: pyautogui.press("nexttrack") if PAG_OK else None,
            "prevtrack":   lambda: pyautogui.press("prevtrack") if PAG_OK else None,
            "showdesktop": lambda: pyautogui.hotkey("win","d") if PAG_OK else None,
            "goback":      lambda: pyautogui.hotkey("alt","left") if PAG_OK else None,
            "goforward":   lambda: pyautogui.hotkey("alt","right") if PAG_OK else None,
        }
        fn = CMD_MAP.get(c)
        if fn:
            try: fn()
            except Exception as e: self.log_fn(f"⚠ {e}")
        else:
            self.log_fn(f"⚠ Unknown system cmd: {c}")



        if not SR_OK:
            self.log_fn("⚠ Voice: pip install SpeechRecognition pyaudio")
            return False
        if self._running:
            return True
        try:
            self._rec = sr.Recognizer()
            self._rec.energy_threshold         = 300
            self._rec.dynamic_energy_threshold = True
            self._rec.pause_threshold          = 0.6
            self._mic = sr.Microphone()
            # Calibrate for ambient noise once
            with self._mic as src:
                self._rec.adjust_for_ambient_noise(src, duration=1)
            self._running = True
            threading.Thread(target=self._loop, daemon=True).start()
            self.log_fn("🎤 Voice started — speak a command!")
            return True
        except Exception as e:
            self.log_fn(f"⚠ Mic error: {e}")
            return False


# ══════════════════════════════════════════════════════════════════
#  ARROW KEY HELPERS
# ══════════════════════════════════════════════════════════════════
VK = {"left":0x25, "up":0x26, "right":0x27, "down":0x28}

def press_arrow(direction):
    if CTYPES_OK:
        try:
            vk = VK[direction]
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.02)
            ctypes.windll.user32.keybd_event(vk, 0, 0x0002, 0)
            return True
        except Exception: pass
    if PAG_OK:
        try: pyautogui.press(direction); return True
        except Exception: pass
    return False

def hold_arrow(direction):
    if CTYPES_OK:
        try: ctypes.windll.user32.keybd_event(VK[direction], 0, 0, 0); return True
        except Exception: pass
    return False

def release_arrow(direction):
    if CTYPES_OK:
        try: ctypes.windll.user32.keybd_event(VK[direction], 0, 0x0002, 0); return True
        except Exception: pass
    return False


# ══════════════════════════════════════════════════════════════════
#  CAMERA ENGINE
# ══════════════════════════════════════════════════════════════════
CV_C = {
    "PALM":(80,255,100),
    "PALM_LEFT":(255,150,0),  "PALM_RIGHT":(0,200,255),
    "PALM_UP":(80,255,100),   "PALM_DOWN":(0,150,255),
    "THREE":(178,72,236),
    "SCROLL_UP":(80,255,100), "SCROLL_DOWN":(0,150,255),
    "IDLE":(120,120,140),
}

GESTURE_LABEL = {
    "PALM_LEFT":"<- LEFT",    "PALM_RIGHT":"-> RIGHT",
    "PALM_UP":"^ UP",         "PALM_DOWN":"v DOWN",
    "PALM_LEFT2":"HOLD <-",   "PALM_RIGHT2":"HOLD ->",
    "PALM_UP2":"HOLD ^",      "PALM_DOWN2":"HOLD v",
    "THREE":"3-FINGER",
    "SCROLL_UP":"SCROLL UP",  "SCROLL_DOWN":"SCROLL DOWN",
    "PALM":"PALM",            "IDLE":"...",
}

DIR_MAP = {
    "PALM_UP":"up","PALM_DOWN":"down",
    "PALM_LEFT":"left","PALM_RIGHT":"right"
}


def draw_hud(frame, gesture, fps, lm=None, voice_text="",
             status="", finger_pos=None, click_flash=0, mic_off=False):
    if not CV2_OK: return frame
    h, w = frame.shape[:2]
    c = CV_C.get(gesture, (120,120,140))
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,60), (7,7,20), -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "AirScroll", (14,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w-100,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,255,100), 2)
    if lm:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, lm, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
        wx = int(lm.landmark[0].x * w); wy = int(lm.landmark[0].y * h)
        cv2.circle(frame, (wx,wy), 10, c, -1)
        cv2.circle(frame, (wx,wy), 12, (255,255,255), 2)
    if finger_pos:
        fx, fy = finger_pos
        cv2.circle(frame, (fx,fy), 16, (0,255,255), 2)
        cv2.circle(frame, (fx,fy), 4, (0,255,255), -1)
        cv2.line(frame, (fx-20,fy), (fx+20,fy), (0,255,255), 1)
        cv2.line(frame, (fx,fy-20), (fx,fy+20), (0,255,255), 1)
    if click_flash > 0 and finger_pos:
        fx, fy = finger_pos
        ov2 = frame.copy()
        cv2.circle(ov2, (fx,fy), int(30*click_flash), (255,255,0), -1)
        cv2.addWeighted(ov2, click_flash*0.5, frame, 1-click_flash*0.5, 0, frame)
    label = GESTURE_LABEL.get(gesture, gesture)
    (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    bx, by = 12, h-50
    cv2.rectangle(frame, (bx-8, by-th2-10), (bx+tw+8, by+10), c, -1)
    cv2.putText(frame, label, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 1.0, (10,10,20), 2)
    if voice_text:
        cv2.rectangle(frame, (w-420,h-42), (w-8,h-14), (20,40,20), -1)
        cv2.putText(frame, f"MIC:{voice_text[:40]}", (w-415,h-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80,255,100), 1)
    if status:
        col = (255,80,80) if "HOLD" in status else (255,220,0)
        cv2.putText(frame, status, (w//2-120, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    guides = ["<- LEFT (x2=hold)","-> RIGHT (x2=hold)",
              "^ UP (x2=hold)","v DOWN (x2=hold)",
              "☝ Index=Tap  🤟 3-Fin=Scroll"]
    for i, g in enumerate(guides):
        cv2.putText(frame, g, (w-230, 85+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,150), 1)
    return frame




# ══════════════════════════════════════════════════════════════════
#  AD SKIPPER — DOM-based detection via Selenium
# ══════════════════════════════════════════════════════════════════
class AdSkipper:
    """
    Attaches to Chrome (launched with --remote-debugging-port=9222)
    and skips YouTube ads via DOM manipulation.

    Strategy (in order):
    1. JS injection on main frame — click any visible skip button
    2. Switch into YouTube iframe if needed — try again
    3. Keyboard fallback — Tab+Enter to focus+click skip button
    4. If unskippable short ad — press End key to jump to end
    """

    def __init__(self, log_fn=None):
        self._driver = None
        self._lock   = threading.Lock()
        self._log    = log_fn or print

    # ── JS that finds and clicks skip button in current frame ─────
    _JS_SKIP = """
    (function() {
        var selectors = [
            '.ytp-ad-skip-button',
            '.ytp-ad-skip-button-modern',
            '.ytp-ad-skip-button-slot button',
            'button.ytp-ad-skip-button',
            '[class*="ytp-ad-skip-button"]',
            '[class*="skip-ad"]',
            '[id*="skip"]'
        ];
        for (var s of selectors) {
            var els = document.querySelectorAll(s);
            for (var el of els) {
                var r = el.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) {
                    el.click();
                    el.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
                    return 'clicked:' + s;
                }
            }
        }
        return 'not_found';
    })();
    """

    # ── JS to detect if ad is currently playing ───────────────────
    _JS_AD_CHECK = """
    (function() {
        var indicators = [
            '.ad-showing',
            '.ytp-ad-player-overlay',
            '.ytp-ad-player-overlay-instream-info',
            '.video-ads.ytp-ad-module',
            '.ytp-ad-badge',
            '[class*="ad-showing"]'
        ];
        for (var s of indicators) {
            if (document.querySelector(s)) return 'ad:' + s;
        }
        // Check video title for ad marker
        var title = document.querySelector('.ytp-ad-preview-text');
        if (title) return 'ad:preview';
        return 'no_ad';
    })();
    """

    def connect(self):
        if not SELENIUM_OK:
            self._log("⚠ pip install selenium webdriver-manager")
            return False
        try:
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service as CS
            opts = Options()
            # When using debuggerAddress, cannot use excludeSwitches
            opts.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                svc = CS(ChromeDriverManager().install())
            except Exception:
                svc = CS()
            try:
                self._driver = webdriver.Chrome(service=svc, options=opts)
            except Exception:
                self._driver = webdriver.Chrome(options=opts)
            self._log("🌐 Selenium → Chrome connected")
            return True
        except Exception as e:
            self._log(f"⚠ Chrome connect failed: {e}")
            self._driver = None
            return False

    def skip_ad(self):
        """
        Try every method to skip the ad.
        Returns: 'skipped' | 'no_skip_btn' | 'no_ad' | 'error'
        """
        if not self._driver:
            return "error"

        with self._lock:
            try:
                # ── Strategy 1: JS in main frame ─────────────────
                result = self._try_js_skip_main()
                if result == "skipped":
                    return "skipped"

                # ── Strategy 2: Switch into iframes and try again ─
                result = self._try_js_skip_iframes()
                if result == "skipped":
                    return "skipped"

                # ── Detect if ad is playing at all ────────────────
                ad_result = self._driver.execute_script(self._JS_AD_CHECK)
                ad_playing = ad_result and ad_result.startswith("ad:")

                if ad_playing:
                    # ── Strategy 3: Keyboard Tab+Enter ───────────
                    try:
                        if PAG_OK:
                            pyautogui.hotkey("shift","tab")
                            time.sleep(0.1)
                            pyautogui.press("tab")
                            time.sleep(0.15)
                            pyautogui.press("enter")
                            time.sleep(0.3)
                            ad_after = self._driver.execute_script(self._JS_AD_CHECK)
                            if ad_after == "no_ad":
                                return "skipped"
                    except Exception:
                        pass

                    # ── Strategy 4: JS via Chrome address bar ─────
                    try:
                        if PAG_OK:
                            js = (
                                "javascript:("
                                "function(){"
                                "var s=['.ytp-ad-skip-button','.ytp-ad-skip-button-modern','[class*=ytp-ad-skip]'];"
                                "for(var i of s){var e=document.querySelector(i);if(e){e.click();return;}}"
                                "})()"
                            )
                            pyautogui.hotkey("ctrl", "l")
                            time.sleep(0.2)
                            pyautogui.hotkey("ctrl", "a")
                            pyautogui.typewrite(js, interval=0.01)
                            time.sleep(0.1)
                            pyautogui.press("enter")
                            time.sleep(0.3)
                            ad_after2 = self._driver.execute_script(self._JS_AD_CHECK)
                            if ad_after2 == "no_ad":
                                return "skipped"
                    except Exception:
                        pass

                    # ── Strategy 5: Press End to skip short unskippable ad
                    try:
                        if PAG_OK:
                            pyautogui.press("end")
                    except Exception:
                        pass
                    return "no_skip_btn"

                return "no_ad"

            except Exception as e:
                err = str(e).lower()
                if "disconnected" in err or "no such window" in err:
                    self._driver = None
                    return "error"
                return "error"

    def _try_js_skip_main(self):
        """Run skip JS in current (main) frame."""
        try:
            self._driver.switch_to.default_content()
            result = self._driver.execute_script(self._JS_SKIP)
            print(f"[AdSkip] main frame: {result}", flush=True)
            if result and result.startswith("clicked:"):
                return "skipped"
        except Exception as e:
            print(f"[AdSkip] main frame err: {e}", flush=True)
        return "not_found"

    def _try_js_skip_iframes(self):
        """Switch into each iframe and try the JS skip."""
        try:
            self._driver.switch_to.default_content()
            frames = self._driver.find_elements(By.TAG_NAME, "iframe")
            print(f"[AdSkip] checking {len(frames)} iframes", flush=True)
            for i, frame in enumerate(frames):
                try:
                    self._driver.switch_to.frame(frame)
                    result = self._driver.execute_script(self._JS_SKIP)
                    print(f"[AdSkip] iframe[{i}]: {result}", flush=True)
                    if result and result.startswith("clicked:"):
                        self._driver.switch_to.default_content()
                        return "skipped"
                    self._driver.switch_to.default_content()
                except Exception:
                    try: self._driver.switch_to.default_content()
                    except Exception: pass
        except Exception as e:
            print(f"[AdSkip] iframe scan err: {e}", flush=True)
        return "not_found"

    def is_ad_playing(self):
        """Quick check — True if ad DOM indicator found."""
        if not self._driver: return False
        try:
            self._driver.switch_to.default_content()
            r = self._driver.execute_script(self._JS_AD_CHECK)
            return r and r.startswith("ad:")
        except Exception:
            return False

    def disconnect(self):
        self._driver = None


# Global ad skipper instance (shared across engine runs)
_ad_skipper = AdSkipper()


def _ensure_ad_skipper_connected(log_fn=None):
    """Connect ad skipper lazily — only when needed."""
    global _ad_skipper
    if _ad_skipper._driver is None:
        _ad_skipper._log = log_fn or _ad_skipper._log
        _ad_skipper.connect()


def run_engine(cam_idx, log_fn, stop_flag):
    if not CV2_OK:
        log_fn("ERROR: cv2/mediapipe not installed"); return
    cap = None
    try:
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        clf = PalmClassifier()
        scroll_clf = ThreeFingerScroll()
        pinch_clf  = DoublePinchDetector()
        DOUBLE_WINDOW = 1.2
        last_swipe_dir = None; last_swipe_t = 0.0
        held_key = None; click_flash = 0.0
        dwell_x = -1.0; dwell_y = -1.0   # -1 = no position yet
        dwell_t = 0.0;  dwell_fired = False
        OPPOSITE = {"up":"down","down":"up","left":"right","right":"left"}
        last_fire = {}
        COOLDOWN = {"PALM_LEFT":0.4,"PALM_RIGHT":0.4,"PALM_UP":0.4,"PALM_DOWN":0.4,
                    "SCROLL_UP":0.0,"SCROLL_DOWN":0.0}
        status = ""; sc = 0; prev_t = time.time()
        err_count = 0; voice_text = ""; voice_clear = 0
        log_fn("Camera opened")
        with mp.solutions.hands.Hands(max_num_hands=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75, model_complexity=1) as hands:
            while not stop_flag[0]:
                try:
                    ok, frame = cap.read()
                    if not ok:
                        err_count += 1
                        if err_count >= 10:
                            log_fn("Camera lost — retrying...")
                            cap.release(); time.sleep(1)
                            cap = cv2.VideoCapture(cam_idx)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            err_count = 0
                        continue
                    err_count = 0
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    now = time.time(); cg = "IDLE"; lm = None
                    if res.multi_hand_landmarks:
                        lm = res.multi_hand_landmarks[0]
                        if lm:
                            th, ix, mi, ri, pi = _fingers_up(lm)
                            # ☝ INDEX ONLY  → tap/dwell mode
                            # 🤟 THREE FINGERS → scroll mode (handled by scroll_clf)
                            index_only_mode  = (not th) and ix and (not mi) and (not ri) and (not pi)
                            three_finger_mode = th and ix and mi and (not ri) and (not pi)

                            vp = stop_flag[2] if len(stop_flag)>2 else None
                            rp_dwell = getattr(vp, "_mirror", None) if vp else None

                            if index_only_mode:
                                # ── Dwell tap — index finger still 1.2s ──────
                                nx = lm.landmark[8].x
                                ny = lm.landmark[8].y
                                RESET_X = 80.0 / max(w, 1)
                                RESET_Y = 80.0 / max(h, 1)

                                if dwell_x < 0:
                                    dwell_x = nx; dwell_y = ny
                                    dwell_t = now; dwell_fired = False
                                elif abs(nx - dwell_x) > RESET_X or abs(ny - dwell_y) > RESET_Y:
                                    dwell_x = nx; dwell_y = ny
                                    dwell_t = now; dwell_fired = False
                                else:
                                    held = now - dwell_t
                                    progress = min(held / 1.2, 1.0)
                                    if rp_dwell:
                                        rp_dwell.show_dwell(nx, ny, progress)

                                    if held >= 1.2 and not dwell_fired:
                                        dwell_fired = True
                                        if PAG_OK:
                                            try:
                                                sw2, sh2 = pyautogui.size()
                                                rx = max(0, min(int(nx*sw2), sw2-1))
                                                ry = max(0, min(int(ny*sh2), sh2-1))
                                                pyautogui.click(rx, ry)
                                                click_flash = 1.0
                                                log_fn(f"👆 Tap ({rx},{ry})")
                                                status = "👆 TAP"; sc = now + 1.5
                                            except Exception as e:
                                                log_fn(f"Tap err:{e}")
                                        if rp_dwell:
                                            rp_dwell.show_tap(nx, ny)
                                        def _rd():
                                            nonlocal dwell_fired,dwell_x,dwell_y
                                            dwell_fired=False; dwell_x=-1.0; dwell_y=-1.0
                                        import threading as _t; _t.Timer(1.5,_rd).start()
                            else:
                                # Not in tap mode — reset dwell
                                dwell_x = -1.0; dwell_y = -1.0; dwell_fired = False
                                if rp_dwell: rp_dwell.show_dwell(0, 0, 0.0)

                        else:
                            dwell_x = -1.0; dwell_y = -1.0; dwell_fired = False

                        sg = scroll_clf.classify(lm)
                        cg = sg if sg in ("SCROLL_UP","SCROLL_DOWN","THREE") else clf.classify(lm)
                        if cg in ("SCROLL_UP","SCROLL_DOWN"):
                            if (now - last_fire.get(cg,0)) >= COOLDOWN[cg]:
                                last_fire[cg] = now
                                if PAG_OK:
                                    try: pyautogui.scroll(80 if cg=="SCROLL_UP" else -80)
                                    except Exception: pass
                                status = GESTURE_LABEL[cg]; sc = now + 0.4
                        elif cg == "PALM":
                            if (now - last_fire.get("PALM_RESUME",0)) >= 2.0:
                                vref = stop_flag[2] if len(stop_flag) > 2 else None
                                if vref and getattr(vref,"_mic_paused",False):
                                    last_fire["PALM_RESUME"] = now
                                    def _palm_skip(lf=log_fn):
                                        # Reconnect if needed
                                        if _ad_skipper._driver is None:
                                            lf("🔌 Connecting to Chrome...")
                                            ok = _ad_skipper.connect()
                                            if not ok:
                                                lf("⚠ Chrome not found — open YouTube via AirScroll voice")
                                                # Pure keyboard fallback
                                                if PAG_OK:
                                                    pyautogui.press("tab")
                                                    time.sleep(0.2)
                                                    pyautogui.press("enter")
                                                return
                                        lf("🔍 Checking for ad...")
                                        result = _ad_skipper.skip_ad()
                                        lf(f"📺 Skip result: {result}")
                                        if result == "skipped":
                                            lf("⏭ Ad skipped!")
                                        elif result == "no_skip_btn":
                                            lf("⏳ Ad not skippable yet")
                                        elif result == "no_ad":
                                            lf("▶ No ad — resuming")
                                        else:
                                            lf("⚠ DOM skip failed — pip install selenium webdriver-manager")
                                    threading.Thread(target=_palm_skip, daemon=True).start()
                                    status = "⏭ Skip Ad..."; sc = now + 2
                                    vref.resume_mic()
                                    # Clear browser context so next command is normal NLU
                                    vref._browser_ctx = None
                                    log_fn("🎤 Mic resumed — say any command")
                        elif cg in DIR_MAP:
                            # ── Skip/seek when video playing (mic paused) ──
                            vref_skip = stop_flag[2] if len(stop_flag) > 2 else None
                            if vref_skip and getattr(vref_skip,"_mic_paused",False):
                                direction_skip = DIR_MAP[cg]
                                if (now - last_fire.get("SKIP_GESTURE",0)) >= 1.0:
                                    last_fire["SKIP_GESTURE"] = now
                                    if direction_skip == "right":
                                        # PALM RIGHT = DOM-based Skip Ad
                                        def _dom_skip(lf=log_fn):
                                            _ensure_ad_skipper_connected(lf)
                                            result = _ad_skipper.skip_ad()
                                            if result == "skipped":
                                                lf("⏭ Ad skipped (DOM)!")
                                            elif result == "no_skip_btn":
                                                lf("⏳ Ad playing — no skip button yet")
                                                # Fallback: +10s to skip past short ad
                                                if PAG_OK:
                                                    try:
                                                        import pyautogui as _pag
                                                        for _ in range(3): _pag.press("l")
                                                    except Exception: pass
                                            elif result == "no_ad":
                                                # Normal forward seek
                                                if PAG_OK:
                                                    try:
                                                        import pyautogui as _pag
                                                        _pag.press("l")
                                                    except Exception: pass
                                                lf("⏩ +10s")
                                            else:
                                                lf("⚠ Selenium not connected — install: pip install selenium")
                                        threading.Thread(target=_dom_skip, daemon=True).start()
                                        status = "⏭ Skip Ad"; sc = now + 1.5
                                    elif PAG_OK:
                                        try:
                                            if direction_skip == "up":
                                                pyautogui.press("k")
                                                log_fn("⏯ Play/Pause")
                                                status = "⏯ Play/Pause"; sc = now + 1.0
                                            elif direction_skip == "left":
                                                pyautogui.press("j")
                                                log_fn("⏪ -10s")
                                                status = "⏪ -10s"; sc = now + 1.0
                                            elif direction_skip == "down":
                                                pyautogui.press("k")
                                                log_fn("⏯ Play/Pause")
                                                status = "⏯ Play/Pause"; sc = now + 1.0
                                        except Exception: pass
                                continue  # don't process as normal arrow key
                        if cg in DIR_MAP:
                            cd = COOLDOWN.get(cg, 0.4)
                            if (now - last_fire.get(cg,0)) >= cd:
                                last_fire[cg] = now
                                direction = DIR_MAP[cg]
                                is_double = (direction == last_swipe_dir and
                                             (now - last_swipe_t) <= DOUBLE_WINDOW)
                                if is_double:
                                    if held_key != direction:
                                        if held_key: release_arrow(held_key)
                                        held_key = direction
                                        hold_arrow(direction)
                                        log_fn(f"HOLD {direction.upper()} ON")
                                        status = f"HOLD {direction.upper()}"; sc = now+999
                                    last_swipe_dir = None; last_swipe_t = 0.0
                                else:
                                    if held_key and OPPOSITE.get(held_key) == direction:
                                        release_arrow(held_key)
                                        log_fn(f"HOLD {held_key.upper()} OFF")
                                        status = "Released"; sc = now+1.0
                                        held_key = None
                                        last_swipe_dir = None; last_swipe_t = 0.0
                                    else:
                                        press_arrow(direction)
                                        log_fn(f"Swipe {GESTURE_LABEL[cg]}")
                                        status = GESTURE_LABEL[cg]; sc = now+1.0
                                        last_swipe_dir = direction; last_swipe_t = now
                    if len(stop_flag) > 1 and stop_flag[1]:
                        voice_text = stop_flag[1]; voice_clear = now+3; stop_flag[1] = ""
                    if voice_text and now > voice_clear: voice_text = ""
                    if status and now > sc: status = ""
                    fps = 1.0 / max(now-prev_t, 0.001); prev_t = now
                    f_pos = None
                    if lm:
                        nx = lm.landmark[8].x
                        ny = lm.landmark[8].y
                        f_pos = (int(nx*w), int(ny*h))
                        # Show finger cursor on mirror ONLY in index-finger tap mode
                        th2, ix2, mi2, ri2, pi2 = _fingers_up(lm)
                        is_tap_mode = (not th2) and ix2 and (not mi2) and (not ri2) and (not pi2)
                        voice_ref = stop_flag[2] if len(stop_flag) > 2 else None
                        rp = getattr(voice_ref, "_mirror", None) if voice_ref else None
                        if rp:
                            if is_tap_mode:
                                rp.update_finger(nx, ny)
                            else:
                                rp.hide_finger()

                    elif len(stop_flag) > 2 and stop_flag[2]:
                        v = stop_flag[2]
                        rp2 = getattr(v, "_mirror", None)
                        if rp2:
                            rp2.hide_finger()
                            rp2.show_dwell(-1, -1, 0.0)
                        dwell_x = -1.0; dwell_y = -1.0
                        dwell_t = now;  dwell_fired = False
                    if click_flash > 0: click_flash = max(0.0, click_flash-0.08)
                    # Get mic state for HUD
                    vr = stop_flag[2] if len(stop_flag)>2 else None
                    mic_off = getattr(vr,"_mic_paused",False) if vr else False
                    frame = draw_hud(frame, cg, fps, lm, voice_text, status, f_pos, click_flash, mic_off)
                    cv2.imshow("AirScroll (ESC to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == 27 or stop_flag[0]: break
                except KeyboardInterrupt: break
                except cv2.error as e: log_fn(f"Camera err: {e}"); time.sleep(0.05)
                except Exception as e: log_fn(f"Frame error: {e}"); time.sleep(0.05)
    except Exception as e:
        log_fn(f"Engine error: {e}")
    finally:
        try:
            if held_key: release_arrow(held_key)
        except Exception: pass
        try:
            if cap: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        log_fn("Engine stopped")



class MirrorPanel:
    """Lightweight floating screen mirror — no results list."""
    def __init__(self, root):
        self._root  = root
        self._win   = None
        self._photo = None
        self._canvas= None
        self._sw    = 1920
        self._sh    = 1080
        self._running = False
        self._finger_x = -1.0
        self._finger_y = -1.0
        self._win_geo  = None
        self._dwell_progress = 0.0
        self._create()
        self._root.after(200, self._poll_geo)

    def _create(self):
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        pw, ph = min(sw, 520), min(sh, 680)
        w = tk.Toplevel(self._root)
        self._win = w
        w.title("AirScroll Mirror")
        w.geometry(f"{pw}x{ph}+{sw-pw-4}+30")
        w.configure(bg="#000")
        w.attributes("-topmost", True)
        w.attributes("-alpha", 0.88)
        w.resizable(True, True)
        w.protocol("WM_DELETE_WINDOW", self.hide)

        hdr = tk.Frame(w, bg="#1a1a2e", height=32)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text=" 🖥  AirScroll Mirror",
                 font=("Courier",9,"bold"), fg=GOLD, bg="#1a1a2e"
                 ).pack(side="left", padx=6, pady=5)
        tk.Label(hdr, text="opacity:",
                 font=("Outfit",8), fg=MUTED2, bg="#1a1a2e").pack(side="left", padx=(10,2))
        self._alpha = tk.DoubleVar(value=0.88)
        tk.Scale(hdr, variable=self._alpha, from_=0.3, to=1.0,
                 resolution=0.05, orient="horizontal", length=80,
                 bg="#1a1a2e", fg=MUTED2, troughcolor=CARD2,
                 highlightthickness=0, bd=0, showvalue=False,
                 command=lambda v: w.attributes("-alpha", float(v))
                 ).pack(side="left")
        tk.Button(hdr, text="✕", font=("Courier",9,"bold"),
                  bg="#1a1a2e", fg=GOLD, relief="flat", cursor="hand2",
                  command=self.hide).pack(side="right", padx=6)

        self._status = tk.StringVar(value="Loading...")
        tk.Label(w, textvariable=self._status,
                 font=("Outfit",7), fg=MUTED2, bg="#0a0a1a",
                 anchor="w", padx=8, pady=2).pack(fill="x")

        self._canvas = tk.Canvas(w, bg="#111", highlightthickness=0,
                                  cursor="crosshair")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>",        self._on_click)
        self._canvas.bind("<Double-Button-1>", self._on_dclick)

        tk.Label(w, text="3-finger still 1.2s = tap  |  click = move cursor  |  dbl-click = click",
                 font=("Outfit",7), fg=MUTED, bg="#111").pack(side="bottom", pady=2)

        w.withdraw()  # hidden by default

    def show(self):
        try:
            if self._win: self._win.deiconify(); self._win.lift()
            if not self._running: self._running = True
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception: pass

    def hide(self):
        try:
            if self._win: self._win.withdraw()
        except Exception: pass

    def is_visible(self):
        try: return self._win and self._win.state() != "withdrawn"
        except: return False

    def _loop(self):
        while self._running:
            try: self._capture()
            except Exception: pass
            time.sleep(0.4)

    def _capture(self):
        from PIL import ImageGrab, Image, ImageTk
        shot = ImageGrab.grab()
        self._sw, self._sh = shot.size
        try:
            cw = max(self._canvas.winfo_width(), 10)
            ch = max(self._canvas.winfo_height(), 10)
        except Exception:
            cw, ch = 400, 500
        ratio = min(cw/self._sw, ch/self._sh)
        nw = max(int(self._sw*ratio),1); nh = max(int(self._sh*ratio),1)
        img   = shot.resize((nw,nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        def _apply(p=photo, w=nw, h=nh, cw=cw, ch=ch):
            try:
                ox=(cw-w)//2; oy=(ch-h)//2
                self._canvas.delete("mirror")
                self._canvas.create_image(ox,oy,anchor="nw",image=p,tags="mirror")
                self._photo = p
                self._draw_cursor(ox,oy,w,h)
                self._status.set(f"  {self._sw}x{self._sh}  |  3-finger still=tap")
                try:
                    if self._win and self._win.state()!="withdrawn":
                        self._win_geo=(self._win.winfo_rootx(),self._win.winfo_rooty(),
                                       self._win.winfo_width(),self._win.winfo_height())
                    else: self._win_geo=None
                except Exception: self._win_geo=None
            except Exception: pass
        self._root.after(0, _apply)

    def _draw_cursor(self, ox, oy, iw, ih):
        self._canvas.delete("cursor")
        if self._finger_x < 0: return
        cx = ox + int(self._finger_x * iw)
        cy = oy + int(self._finger_y * ih)
        r   = 14
        prog = self._dwell_progress

        # Dwell progress arc
        if prog > 0:
            ar = r + 6
            self._canvas.create_oval(cx-ar,cy-ar,cx+ar,cy+ar,
                                      outline="#333",width=3,tags="cursor")
            arc_col = "#00ff88" if prog < 0.5 else (GOLD if prog < 0.85 else "#ff4444")
            self._canvas.create_arc(cx-ar,cy-ar,cx+ar,cy+ar,
                                     start=90,extent=-(prog*360),
                                     outline=arc_col,width=3,
                                     style="arc",tags="cursor")
            self._canvas.create_text(cx,cy+ar+10,
                                      text=f"{int(prog*100)}%",
                                      fill=arc_col,font=("Courier",7,"bold"),
                                      tags="cursor")

        # Crosshair
        col = GOLD if prog > 0.85 else CYAN
        self._canvas.create_oval(cx-r,cy-r,cx+r,cy+r,outline=col,width=2,tags="cursor")
        self._canvas.create_oval(cx-3,cy-3,cx+3,cy+3,fill=col,outline="",tags="cursor")
        self._canvas.create_line(cx-r-4,cy,cx-r+2,cy,fill=col,width=1,tags="cursor")
        self._canvas.create_line(cx+r-2,cy,cx+r+4,cy,fill=col,width=1,tags="cursor")
        self._canvas.create_line(cx,cy-r-4,cx,cy-r+2,fill=col,width=1,tags="cursor")
        self._canvas.create_line(cx,cy+r-2,cx,cy+r+4,fill=col,width=1,tags="cursor")

    def update_finger(self, nx, ny):
        self._finger_x = nx; self._finger_y = ny

    def hide_finger(self):
        self._finger_x = -1.0; self._finger_y = -1.0
        self._dwell_progress = 0.0

    def show_dwell(self, nx, ny, progress):
        """Update dwell progress arc (0.0–1.0). Thread-safe."""
        self._dwell_progress = max(0.0, min(1.0, progress))
        self._root.after(0, self._redraw_cursor)

    def _redraw_cursor(self):
        """Immediate cursor redraw without waiting for next screenshot."""
        if not self._canvas: return
        try:
            cw = max(self._canvas.winfo_width(), 1)
            ch = max(self._canvas.winfo_height(), 1)
            ratio = min(cw/self._sw, ch/self._sh)
            iw = max(int(self._sw*ratio), 1)
            ih = max(int(self._sh*ratio), 1)
            ox = (cw-iw)//2; oy = (ch-ih)//2
            self._draw_cursor(ox, oy, iw, ih)
        except Exception: pass

    def show_tap(self, nx, ny):
        self._root.after(0, lambda: self._tap_anim(nx, ny, 0))

    def _tap_anim(self, nx, ny, step):
        if step > 8:
            try: self._canvas.delete("tap")
            except Exception: pass
            return
        try:
            cw=max(self._canvas.winfo_width(),1)
            ch=max(self._canvas.winfo_height(),1)
            ratio=min(cw/self._sw,ch/self._sh)
            iw=int(self._sw*ratio); ih=int(self._sh*ratio)
            ox=(cw-iw)//2; oy=(ch-ih)//2
            cx=ox+int(nx*iw); cy=oy+int(ny*ih)
            self._canvas.delete("tap")
            r=8+step*6
            self._canvas.create_oval(cx-r,cy-r,cx+r,cy+r,
                                      outline=GOLD,width=max(1,3-step//3),tags="tap")
            if step<5:
                ri=max(2,14-step*2)
                self._canvas.create_oval(cx-ri,cy-ri,cx+ri,cy+ri,
                                          fill=GOLD,outline="",tags="tap")
            if step<4:
                self._canvas.create_text(cx,cy-r-8,text="TAP",fill=GOLD,
                                          font=("Courier",9,"bold"),tags="tap")
            self._root.after(40, lambda: self._tap_anim(nx,ny,step+1))
        except Exception: pass

    def _poll_geo(self):
        try:
            if self._win and self._win.state()!="withdrawn":
                self._win_geo=(self._win.winfo_rootx(),self._win.winfo_rooty(),
                               self._win.winfo_width(),self._win.winfo_height())
            else: self._win_geo=None
        except Exception: self._win_geo=None
        try: self._root.after(200, self._poll_geo)
        except Exception: pass

    def _on_click(self, event):
        if not PAG_OK: return
        rx,ry=self._map(event.x,event.y)
        try: pyautogui.moveTo(rx,ry,duration=0); self._status.set(f"  Cursor → ({rx},{ry})")
        except Exception: pass

    def _on_dclick(self, event):
        if not PAG_OK: return
        rx,ry=self._map(event.x,event.y)
        try: pyautogui.click(rx,ry); self._status.set(f"  Clicked ({rx},{ry})")
        except Exception: pass

    def _map(self,mx,my):
        cw=max(self._canvas.winfo_width(),1); ch=max(self._canvas.winfo_height(),1)
        ratio=min(cw/self._sw,ch/self._sh)
        iw=int(self._sw*ratio); ih=int(self._sh*ratio)
        ox=(cw-iw)//2; oy=(ch-ih)//2
        px=max(0.0,min(1.0,(mx-ox)/max(iw,1)))
        py=max(0.0,min(1.0,(my-oy)/max(ih,1)))
        return int(px*self._sw),int(py*self._sh)

    def get_geo(self):
        return self._win_geo


# ══════════════════════════════════════════════════════════════════
#  PHONE REMOTE CONTROL SERVER
# ══════════════════════════════════════════════════════════════════

PHONE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>AirScroll Remote</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{background:#07071A;color:#E2E2F5;font-family:'Courier New',monospace;padding-bottom:30px}
.hdr{background:#0C0C24;padding:12px 14px;display:flex;align-items:center;gap:10px;border-bottom:2px solid #F5C400;position:sticky;top:0;z-index:200}
.logo{width:34px;height:34px;background:#F5C400;display:flex;align-items:center;justify-content:center;font-size:18px;border-radius:6px;flex-shrink:0}
.title{font-size:16px;font-weight:bold;color:#F5C400}
.sub{font-size:9px;color:#6A6A9A}
.stat{margin-left:auto;padding:4px 10px;border-radius:12px;font-size:11px;font-weight:bold;background:#00F573;color:#07071A}
.stat.off{background:#FF3D5A;color:#fff}

/* Tabs */
.tabs{display:flex;background:#0C0C24;border-bottom:1px solid #1E1E45;overflow-x:auto;position:sticky;top:58px;z-index:100}
.tab{flex:1;padding:10px 4px;text-align:center;font-size:11px;color:#6A6A9A;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;min-width:60px}
.tab.active{color:#F5C400;border-bottom-color:#F5C400}

/* Panels */
.panel{display:none;padding:0 0 8px}
.panel.active{display:block}

.sec{padding:10px 14px 4px;font-size:9px;color:#4A4A7A;letter-spacing:1px;text-transform:uppercase}
.grid{display:grid;gap:6px;padding:0 10px 6px}
.g2{grid-template-columns:1fr 1fr}
.g3{grid-template-columns:1fr 1fr 1fr}
.g4{grid-template-columns:1fr 1fr 1fr 1fr}
.g1{grid-template-columns:1fr}
.btn{background:#10102A;border:1px solid #1E1E45;border-radius:10px;padding:12px 6px;text-align:center;cursor:pointer;user-select:none;transition:transform .1s}
.btn:active{transform:scale(.93);border-color:#F5C400}
.icon{font-size:20px;display:block;margin-bottom:3px}
.lbl{font-size:10px;color:#6A6A9A}
.gold{border-color:#F5C400;background:#141000}
.cyan{border-color:#00DCFF;background:#001015}
.green{border-color:#00F573;background:#001008}
.red{border-color:#FF3D5A;background:#140008}
.purple{border-color:#A855F7;background:#0a0015}
.blue{border-color:#3B82F6;background:#000d1a}

/* D-pad */
.dpad{display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr 1fr;gap:6px;padding:6px 10px;max-width:240px;margin:0 auto}
.dpad .btn{padding:16px 6px}
.dpad .mid{background:#161636;border-color:#00DCFF}
.dpad .empty{background:transparent;border:none}

/* Keyboard / typing */
.type-area{margin:6px 10px;display:flex;flex-direction:column;gap:6px}
.type-box{background:#10102A;border:1px solid #1E1E45;border-radius:10px;padding:12px;color:#E2E2F5;font-size:15px;width:100%;outline:none;font-family:'Courier New',monospace}
.type-box:focus{border-color:#F5C400}
.type-row{display:flex;gap:6px}
.type-row .btn{flex:1;padding:10px 4px}
.big-send{background:#F5C400;border:none;border-radius:10px;padding:12px 20px;color:#07071A;font-size:16px;font-weight:bold;cursor:pointer;width:100%;margin-top:2px}
.hint{font-size:9px;color:#4A4A7A;padding:2px 10px;text-align:center}

/* Search bars */
.search-row{display:flex;gap:6px;padding:0 10px 6px;align-items:center}
.search-inp{flex:1;background:#10102A;border:1px solid #1E1E45;border-radius:10px;padding:11px 12px;color:#E2E2F5;font-size:14px;outline:none}
.search-inp:focus{border-color:#F5C400}
.search-go{background:#F5C400;border:none;border-radius:10px;padding:11px 14px;color:#07071A;font-weight:bold;font-size:16px;cursor:pointer}

/* Window list */
.win-list{padding:0 10px 6px}
.win-item{background:#10102A;border:1px solid #1E1E45;border-radius:10px;padding:11px 14px;margin-bottom:6px;cursor:pointer;display:flex;align-items:center;gap:10px;transition:border-color .15s}
.win-item:active{border-color:#F5C400;transform:scale(.98)}
.win-icon{font-size:20px;flex-shrink:0}
.win-title{font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.win-btns{display:flex;gap:4px}
.win-btn{background:#161636;border:1px solid #1E1E45;border-radius:6px;padding:5px 8px;font-size:11px;cursor:pointer;color:#E2E2F5}
.win-btn.min{border-color:#F97316;color:#F97316}
.win-btn.max{border-color:#00DCFF;color:#00DCFF}
.win-btn.cls{border-color:#FF3D5A;color:#FF3D5A}
.refresh-btn{width:100%;padding:10px;background:#161636;border:1px solid #1E1E45;border-radius:10px;color:#F5C400;font-size:13px;cursor:pointer;margin-bottom:8px;font-family:'Courier New',monospace}

/* Log */
.log-box{margin:0 10px;background:#10102A;border:1px solid #1E1E45;border-radius:10px;padding:8px 10px;font-size:10px;color:#6A6A9A;height:90px;overflow-y:auto;line-height:1.6}
</style>
</head>
<body>
<div class="hdr">
  <div class="logo">✋</div>
  <div><div class="title">AirScroll</div><div class="sub">PHONE REMOTE</div></div>
  <div class="stat" id="stat">LIVE</div>
</div>

<!-- ── Tab Bar ── -->
<div class="tabs">
  <div class="tab active" onclick="tab(this,'t-ctrl')">🎮 Control</div>
  <div class="tab" onclick="tab(this,'t-type')">⌨ Type</div>
  <div class="tab" onclick="tab(this,'t-search')">🔍 Search</div>
  <div class="tab" onclick="tab(this,'t-win')">🪟 Windows</div>
  <div class="tab" onclick="tab(this,'t-sys')">⚙ System</div>
</div>

<!-- ══════════════ TAB 1: CONTROL ══════════════ -->
<div class="panel active" id="t-ctrl">

  <div class="sec">🎮 D-PAD</div>
  <div class="dpad">
    <div class="btn empty"></div>
    <div class="btn" onclick="cmd('arrow_up')"><span class="icon">⬆</span><span class="lbl">UP</span></div>
    <div class="btn empty"></div>
    <div class="btn" onclick="cmd('arrow_left')"><span class="icon">⬅</span><span class="lbl">LEFT</span></div>
    <div class="btn mid" onclick="cmd('enter')"><span class="icon">↩</span><span class="lbl">OK</span></div>
    <div class="btn" onclick="cmd('arrow_right')"><span class="icon">➡</span><span class="lbl">RIGHT</span></div>
    <div class="btn empty"></div>
    <div class="btn" onclick="cmd('arrow_down')"><span class="icon">⬇</span><span class="lbl">DOWN</span></div>
    <div class="btn empty"></div>
  </div>

  <div class="grid g3">
    <div class="btn" onclick="cmd('scroll_up')"><span class="icon">🔼</span><span class="lbl">Scroll ↑</span></div>
    <div class="btn" onclick="cmd('tab_key')"><span class="icon">⇥</span><span class="lbl">Tab</span></div>
    <div class="btn" onclick="cmd('scroll_down')"><span class="icon">🔽</span><span class="lbl">Scroll ↓</span></div>
    <div class="btn" onclick="cmd('go_back')"><span class="icon">⬅</span><span class="lbl">Back</span></div>
    <div class="btn" onclick="cmd('escape')"><span class="icon">✕</span><span class="lbl">Escape</span></div>
    <div class="btn" onclick="cmd('go_forward')"><span class="icon">➡</span><span class="lbl">Forward</span></div>
  </div>

  <div class="sec">📺 YOUTUBE</div>
  <div class="grid g3">
    <div class="btn green" onclick="cmd('yt_play_pause')"><span class="icon">⏯</span><span class="lbl">Play/Pause</span></div>
    <div class="btn gold" onclick="cmd('yt_skip_ad')"><span class="icon">⏭</span><span class="lbl">Skip Ad</span></div>
    <div class="btn cyan" onclick="cmd('yt_fullscreen')"><span class="icon">⛶</span><span class="lbl">Fullscreen</span></div>
    <div class="btn" onclick="cmd('yt_back10')"><span class="icon">⏪</span><span class="lbl">-10s</span></div>
    <div class="btn" onclick="cmd('yt_mute')"><span class="icon">🔇</span><span class="lbl">Mute</span></div>
    <div class="btn" onclick="cmd('yt_fwd10')"><span class="icon">⏩</span><span class="lbl">+10s</span></div>
  </div>

  <div class="sec">🔊 VOLUME</div>
  <div class="grid g3">
    <div class="btn red" onclick="cmd('vol_down')"><span class="icon">🔉</span><span class="lbl">VOL -</span></div>
    <div class="btn" onclick="cmd('vol_mute')"><span class="icon">🔇</span><span class="lbl">MUTE</span></div>
    <div class="btn gold" onclick="cmd('vol_up')"><span class="icon">🔊</span><span class="lbl">VOL +</span></div>
  </div>

  <div class="sec">🚀 QUICK APPS</div>
  <div class="grid g3">
    <div class="btn gold" onclick="cmd('open_youtube')"><span class="icon">▶</span><span class="lbl">YouTube</span></div>
    <div class="btn cyan" onclick="cmd('open_chrome')"><span class="icon">🌐</span><span class="lbl">Chrome</span></div>
    <div class="btn" onclick="cmd('open_notepad')"><span class="icon">📝</span><span class="lbl">Notepad</span></div>
    <div class="btn" onclick="cmd('open_excel')"><span class="icon">📊</span><span class="lbl">Excel</span></div>
    <div class="btn" onclick="cmd('open_word')"><span class="icon">📄</span><span class="lbl">Word</span></div>
    <div class="btn purple" onclick="cmd('open_powerpoint')"><span class="icon">📑</span><span class="lbl">PowerPoint</span></div>
  </div>
</div>

<!-- ══════════════ TAB 2: TYPE ══════════════ -->
<div class="panel" id="t-type">
  <div class="sec">⌨ REMOTE TYPING — type here, appears on PC</div>
  <div class="type-area">
    <textarea id="type-inp" class="type-box" rows="3"
      placeholder="Type here → text appears on PC..."></textarea>
    <div class="type-row">
      <div class="btn" onclick="typeKey('backspace')"><span class="icon">⌫</span><span class="lbl">Del</span></div>
      <div class="btn" onclick="typeKey('enter')"><span class="icon">↩</span><span class="lbl">Enter</span></div>
      <div class="btn" onclick="typeKey('tab')"><span class="icon">⇥</span><span class="lbl">Tab</span></div>
      <div class="btn" onclick="typeKey('space')"><span class="icon">⎵</span><span class="lbl">Space</span></div>
      <div class="btn red" onclick="clearType()"><span class="icon">🗑</span><span class="lbl">Clear</span></div>
    </div>
    <button class="big-send" onclick="sendType()">▶ Send to PC</button>
    <div class="hint">Text is sent character by character and typed on PC</div>
  </div>

  <div class="sec">⌨ QUICK SHORTCUTS</div>
  <div class="grid g3">
    <div class="btn" onclick="cmd('copy')"><span class="icon">📋</span><span class="lbl">Copy</span></div>
    <div class="btn" onclick="cmd('paste')"><span class="icon">📌</span><span class="lbl">Paste</span></div>
    <div class="btn" onclick="cmd('cut')"><span class="icon">✂</span><span class="lbl">Cut</span></div>
    <div class="btn" onclick="cmd('undo')"><span class="icon">↩</span><span class="lbl">Undo</span></div>
    <div class="btn" onclick="cmd('redo')"><span class="icon">↪</span><span class="lbl">Redo</span></div>
    <div class="btn" onclick="cmd('save')"><span class="icon">💾</span><span class="lbl">Save</span></div>
    <div class="btn" onclick="cmd('select_all')"><span class="icon">☰</span><span class="lbl">Sel All</span></div>
    <div class="btn" onclick="cmd('find')"><span class="icon">🔍</span><span class="lbl">Find</span></div>
    <div class="btn" onclick="cmd('new_file')"><span class="icon">📄</span><span class="lbl">New</span></div>
  </div>

  <div class="sec">🎤 VOICE COMMAND</div>
  <div class="search-row">
    <input id="vcmd" class="search-inp" type="text" placeholder="open spotify, volume up...">
    <button class="search-go" onclick="sendVoice()">▶</button>
  </div>
</div>

<!-- ══════════════ TAB 3: SEARCH ══════════════ -->
<div class="panel" id="t-search">

  <div class="sec">🔴 YOUTUBE SEARCH</div>
  <div class="search-row">
    <input id="yt-inp" class="search-inp" type="text" placeholder="Search YouTube...">
    <button class="search-go" style="background:#FF0000" onclick="ytSearch()">▶</button>
  </div>
  <div class="grid g2" style="padding:0 10px 6px">
    <div class="btn red" onclick="cmd('yt_focus_search')"><span class="icon">🔍</span><span class="lbl">Focus YT Search Bar</span></div>
    <div class="btn" onclick="cmd('yt_home')"><span class="icon">🏠</span><span class="lbl">YT Home</span></div>
  </div>

  <div class="sec">🔵 GOOGLE SEARCH</div>
  <div class="search-row">
    <input id="gg-inp" class="search-inp" type="text" placeholder="Search Google...">
    <button class="search-go" onclick="ggSearch()">🔍</button>
  </div>
  <div class="grid g2" style="padding:0 10px 6px">
    <div class="btn blue" onclick="cmd('google_focus_search')"><span class="icon">🔍</span><span class="lbl">Focus Address Bar</span></div>
    <div class="btn" onclick="cmd('new_tab')"><span class="icon">＋</span><span class="lbl">New Tab</span></div>
  </div>

  <div class="sec">🌐 OPEN URL</div>
  <div class="search-row">
    <input id="url-inp" class="search-inp" type="url" placeholder="https://...">
    <button class="search-go" onclick="openUrl()">→</button>
  </div>

  <div class="sec">🔖 QUICK SITES</div>
  <div class="grid g3">
    <div class="btn gold" onclick="openSite('https://youtube.com')"><span class="icon">▶</span><span class="lbl">YouTube</span></div>
    <div class="btn blue" onclick="openSite('https://google.com')"><span class="icon">G</span><span class="lbl">Google</span></div>
    <div class="btn" onclick="openSite('https://github.com')"><span class="icon">⚙</span><span class="lbl">GitHub</span></div>
    <div class="btn cyan" onclick="openSite('https://chat.openai.com')"><span class="icon">🤖</span><span class="lbl">ChatGPT</span></div>
    <div class="btn" onclick="openSite('https://whatsapp.com')"><span class="icon">💬</span><span class="lbl">WhatsApp</span></div>
    <div class="btn purple" onclick="openSite('https://instagram.com')"><span class="icon">📸</span><span class="lbl">Instagram</span></div>
  </div>
</div>

<!-- ══════════════ TAB 4: WINDOWS ══════════════ -->
<div class="panel" id="t-win">
  <div class="sec">🪟 OPEN WINDOWS</div>
  <div style="padding:0 10px 4px">
    <button class="refresh-btn" onclick="loadWindows()">⟳ Refresh Window List</button>
  </div>

  <div class="sec">⚡ WINDOW CONTROLS</div>
  <div class="grid g3" style="padding:0 10px 8px">
    <div class="btn" onclick="cmd('win_minimize')"><span class="icon">▁</span><span class="lbl">Minimize</span></div>
    <div class="btn cyan" onclick="cmd('win_maximize')"><span class="icon">□</span><span class="lbl">Maximize</span></div>
    <div class="btn red" onclick="cmd('close_window')"><span class="icon">✕</span><span class="lbl">Close</span></div>
    <div class="btn" onclick="cmd('win_restore')"><span class="icon">⧉</span><span class="lbl">Restore</span></div>
    <div class="btn gold" onclick="cmd('alt_tab')"><span class="icon">⇄</span><span class="lbl">Alt+Tab</span></div>
    <div class="btn purple" onclick="cmd('task_view')"><span class="icon">⊞</span><span class="lbl">Task View</span></div>
  </div>

  <div class="win-list" id="win-list">
    <div style="text-align:center;color:#4A4A7A;padding:20px;font-size:12px">
      Press Refresh to load open windows
    </div>
  </div>
</div>

<!-- ══════════════ TAB 5: SYSTEM ══════════════ -->
<div class="panel" id="t-sys">

  <div class="sec">🖥 DISPLAY</div>
  <div class="grid g2">
    <div class="btn" onclick="cmd('screenshot')"><span class="icon">📸</span><span class="lbl">Screenshot</span></div>
    <div class="btn" onclick="cmd('show_desktop')"><span class="icon">🖥</span><span class="lbl">Show Desktop</span></div>
    <div class="btn cyan" onclick="cmd('win_maximize')"><span class="icon">□</span><span class="lbl">Maximize</span></div>
    <div class="btn" onclick="cmd('win_minimize')"><span class="icon">▁</span><span class="lbl">Minimize All</span></div>
  </div>

  <div class="sec">⚙ SYSTEM</div>
  <div class="grid g2">
    <div class="btn purple" onclick="cmd('task_manager')"><span class="icon">📊</span><span class="lbl">Task Manager</span></div>
    <div class="btn" onclick="cmd('open_settings')"><span class="icon">⚙</span><span class="lbl">Settings</span></div>
    <div class="btn" onclick="cmd('open_explorer')"><span class="icon">📁</span><span class="lbl">Explorer</span></div>
    <div class="btn" onclick="cmd('open_calculator')"><span class="icon">🔢</span><span class="lbl">Calculator</span></div>
    <div class="btn red" onclick="if(confirm('Lock PC?'))cmd('lock')"><span class="icon">🔒</span><span class="lbl">Lock PC</span></div>
    <div class="btn red" onclick="if(confirm('Shutdown?'))cmd('shutdown')"><span class="icon">⏻</span><span class="lbl">Shutdown</span></div>
  </div>

  <div class="sec">📋 LOG</div>
  <div class="log-box" id="log">Waiting...</div>
</div>

<script>
// ── Tab switching ────────────────────────────────────────────────
function tab(el, id) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById(id).classList.add('active');
  if (id === 't-win') loadWindows();
}

// ── API calls ────────────────────────────────────────────────────
async function cmd(action, extra) {
  try {
    const body = extra ? {...extra, action} : {action};
    const r = await fetch('/cmd', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const d = await r.json();
    addLog((d.msg || action));
    return d;
  } catch(e) {
    addLog('Error: '+e.message);
    document.getElementById('stat').className='stat off';
    document.getElementById('stat').textContent='ERR';
  }
}

// ── Remote Typing ────────────────────────────────────────────────
async function sendType() {
  const inp = document.getElementById('type-inp');
  const text = inp.value;
  if (!text) return;
  inp.value = '';
  try {
    const r = await fetch('/type', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const d = await r.json();
    addLog('⌨ Typed: ' + text.substring(0,20) + (text.length>20?'...':''));
  } catch(e) { addLog('Type error: '+e.message); }
}

function typeKey(key) {
  cmd('key_' + key);
}

function clearType() {
  document.getElementById('type-inp').value = '';
}

// ── Search ───────────────────────────────────────────────────────
async function ytSearch() {
  const q = document.getElementById('yt-inp').value.trim();
  if (!q) { cmd('yt_focus_search'); return; }
  document.getElementById('yt-inp').value = '';
  const r = await cmd('yt_search', {query: q});
  addLog('YT Search: ' + q);
}

async function ggSearch() {
  const q = document.getElementById('gg-inp').value.trim();
  if (!q) { cmd('google_focus_search'); return; }
  document.getElementById('gg-inp').value = '';
  await cmd('google_search', {query: q});
}

async function openUrl() {
  const url = document.getElementById('url-inp').value.trim();
  if (!url) return;
  document.getElementById('url-inp').value = '';
  await cmd('open_url', {url});
}

async function openSite(url) {
  await cmd('open_url', {url});
}

// ── Voice command ────────────────────────────────────────────────
async function sendVoice() {
  const inp = document.getElementById('vcmd');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  try {
    const r = await fetch('/voice', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const d = await r.json();
    addLog('🎤 '+text+' → '+(d.msg||'ok'));
  } catch(e) { addLog('Err: '+e.message); }
}
document.addEventListener('keydown', e => {
  if (e.key==='Enter' && document.activeElement.id==='vcmd') sendVoice();
  if (e.key==='Enter' && document.activeElement.id==='type-inp') {}
});

// ── Window list ──────────────────────────────────────────────────
async function loadWindows() {
  const list = document.getElementById('win-list');
  list.innerHTML = '<div style="text-align:center;color:#F5C400;padding:12px">Loading...</div>';
  try {
    const r = await fetch('/windows');
    const d = await r.json();
    if (!d.windows || !d.windows.length) {
      list.innerHTML = '<div style="text-align:center;color:#6A6A9A;padding:12px">No windows found</div>';
      return;
    }
    list.innerHTML = '';
    d.windows.forEach(w => {
      const el = document.createElement('div');
      el.className = 'win-item';
      const icon = getWinIcon(w.title);
      el.innerHTML = `
        <span class="win-icon">${icon}</span>
        <span class="win-title" title="${w.title}">${w.title}</span>
        <div class="win-btns">
          <button class="win-btn min" onclick="focusWin(${w.hwnd});event.stopPropagation()">▁</button>
          <button class="win-btn max" onclick="maxWin(${w.hwnd});event.stopPropagation()">□</button>
          <button class="win-btn cls" onclick="closeWin(${w.hwnd});event.stopPropagation()">✕</button>
        </div>`;
      el.onclick = () => focusWin(w.hwnd);
      list.appendChild(el);
    });
  } catch(e) {
    list.innerHTML = '<div style="text-align:center;color:#FF3D5A;padding:12px">Error loading windows</div>';
  }
}

function getWinIcon(title) {
  const t = title.toLowerCase();
  if (t.includes('youtube') || t.includes('chrome')) return '🌐';
  if (t.includes('excel')) return '📊';
  if (t.includes('word')) return '📄';
  if (t.includes('powerpoint')) return '📑';
  if (t.includes('notepad')) return '📝';
  if (t.includes('explorer')) return '📁';
  if (t.includes('code') || t.includes('pycharm')) return '💻';
  if (t.includes('vlc')) return '🎬';
  if (t.includes('spotify')) return '🎵';
  if (t.includes('task')) return '📊';
  return '🪟';
}

async function focusWin(hwnd) {
  const r = await cmd('win_focus', {hwnd});
  setTimeout(loadWindows, 500);
}
async function maxWin(hwnd) { await cmd('win_maximize_hwnd', {hwnd}); }
async function closeWin(hwnd) {
  if (confirm('Close this window?')) {
    await cmd('win_close_hwnd', {hwnd});
    setTimeout(loadWindows, 600);
  }
}

// ── Log ──────────────────────────────────────────────────────────
function addLog(msg) {
  const log = document.getElementById('log');
  if (!log) return;
  const t = new Date().toLocaleTimeString();
  log.innerHTML = '['+t+'] '+msg+'<br>'+log.innerHTML;
}

// Poll log every 2s
setInterval(async () => {
  try {
    const r = await fetch('/log');
    const d = await r.json();
    if (d.msg) addLog(d.msg);
    document.getElementById('stat').className='stat';
    document.getElementById('stat').textContent='LIVE';
  } catch(e) {
    document.getElementById('stat').className='stat off';
    document.getElementById('stat').textContent='ERR';
  }
}, 2000);
</script>
</body>
</html>"""




class PhoneServer:
    """
    Lightweight Flask server for phone remote control.
    Phone connects via WiFi — no app install needed, just open browser.
    """

    def __init__(self, ui_ref=None, port=5000):
        self._ui      = ui_ref
        self._port    = port
        self._app     = None
        self._thread  = None
        self._running = False
        self._last_log = ""
        self._ip      = self._get_local_ip()

    def _get_local_ip(self):
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start(self):
        if not FLASK_OK:
            if self._ui:
                self._ui._log("⚠ pip install flask  (for phone remote)")
            return False
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if self._ui:
            self._ui._log(f"📱 Phone server: http://{self._ip}:{self._port}")
        return True

    def stop(self):
        self._running = False

    def get_url(self):
        return f"http://{self._ip}:{self._port}"

    def _log(self, msg):
        self._last_log = msg
        if self._ui:
            self._ui._log(f"📱 {msg}")

    def _run(self):
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)   # suppress Flask access logs

        app = Flask(__name__)
        self._app = app

        @app.route("/")
        def index():
            return Response(PHONE_HTML, mimetype="text/html")

        @app.route("/log")
        def get_log():
            msg = self._last_log
            self._last_log = ""
            return jsonify({"msg": msg})

        @app.route("/cmd", methods=["POST"])
        def handle_cmd():
            data = request.get_json(force=True, silent=True) or {}
            action = data.get("action","")
            result = self._execute(action, data)
            return jsonify({"ok":True,"msg":result})

        @app.route("/voice", methods=["POST"])
        def handle_voice():
            data = request.get_json(force=True, silent=True) or {}
            text = data.get("text","").strip()
            if text:
                if self._ui and self._ui._voice:
                    self._ui._voice._handle(text)
                    self._log(f"Voice: {text}")
                    return jsonify({"ok":True,"msg":f"Executed: {text}"})
                else:
                    r = None
                    if self._ui and hasattr(self._ui,'_voice') and self._ui._voice:
                        r = self._ui._voice._local_command_fallback(text)
                    if r:
                        return jsonify({"ok":True,"msg":r})
                    return jsonify({"ok":False,"msg":"Voice engine not running"})
            return jsonify({"ok":False,"msg":"No text"})

        @app.route("/type", methods=["POST"])
        def handle_type():
            data = request.get_json(force=True, silent=True) or {}
            text = data.get("text","")
            if text and PAG_OK:
                try:
                    import pyperclip
                    pyperclip.copy(text)
                    pyautogui.hotkey("ctrl","v")
                    return jsonify({"ok":True,"msg":f"Typed {len(text)} chars via clipboard"})
                except Exception:
                    pass
                try:
                    pyautogui.write(text, interval=0.02)
                    return jsonify({"ok":True,"msg":f"Typed {len(text)} chars"})
                except Exception as e:
                    return jsonify({"ok":False,"msg":str(e)})
            return jsonify({"ok":False,"msg":"Nothing to type"})

        @app.route("/windows")
        def get_windows():
            return jsonify({"windows": self._get_windows()})

        app.run(host="0.0.0.0", port=self._port, threaded=True, use_reloader=False)

    def _get_windows(self):
        """Return list of visible windows with hwnd for focus/close."""
        wins = []
        try:
            import ctypes as _ct
            user32 = _ct.windll.user32
            SKIP = {"","Program Manager","Windows Input Experience",
                    "Microsoft Text Input Application","AirScroll"}

            def enum_cb(hwnd, _):
                if not user32.IsWindowVisible(hwnd): return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length < 2: return True
                buf = _ct.create_unicode_buffer(length+1)
                user32.GetWindowTextW(hwnd, buf, length+1)
                title = buf.value.strip()
                if title and title not in SKIP:
                    wins.append({"hwnd": hwnd, "title": title})
                return True

            WNDENUMPROC = _ct.WINFUNCTYPE(_ct.c_bool, _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int))
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        except Exception as e:
            self._log(f"Window list error: {e}")
        return wins[:30]

    def _focus_window(self, hwnd):
        """Bring a window to foreground by hwnd."""
        try:
            import ctypes as _ct
            user32 = _ct.windll.user32
            user32.ShowWindow(hwnd, 9)   # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False

    def _close_window_hwnd(self, hwnd):
        """Close a window by hwnd."""
        try:
            import ctypes as _ct
            _ct.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            return True
        except Exception:
            return False

    def _maximize_window_hwnd(self, hwnd):
        """Maximize a window by hwnd."""
        try:
            import ctypes as _ct
            _ct.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            return True
        except Exception:
            return False

    def _execute(self, action, data=None):
        """Execute a phone remote command."""
        data = data or {}
        try:
            # ── Arrow keys ──────────────────────────────────────
            if action == "arrow_up":
                if PAG_OK: pyautogui.press("up")
                return "↑ Up"
            if action == "arrow_down":
                if PAG_OK: pyautogui.press("down")
                return "↓ Down"
            if action == "arrow_left":
                if PAG_OK: pyautogui.press("left")
                return "← Left"
            if action == "arrow_right":
                if PAG_OK: pyautogui.press("right")
                return "→ Right"
            if action == "enter":
                if PAG_OK: pyautogui.press("enter")
                return "↩ Enter"
            if action == "escape":
                if PAG_OK: pyautogui.press("escape")
                return "Esc"
            if action == "backspace":
                if PAG_OK: pyautogui.press("backspace")
                return "⌫ Backspace"

            # ── YouTube ─────────────────────────────────────────
            if action == "yt_play_pause":
                if PAG_OK: pyautogui.press("k")
                return "⏯ Play/Pause"
            if action == "yt_skip_ad":
                def _skip():
                    _ensure_ad_skipper_connected(self._log)
                    result = _ad_skipper.skip_ad()
                    self._log(f"Skip ad: {result}")
                threading.Thread(target=_skip, daemon=True).start()
                return "⏭ Checking ad..."
            if action == "yt_fwd10":
                if PAG_OK: pyautogui.press("l")
                return "⏩ +10s"
            if action == "yt_back10":
                if PAG_OK: pyautogui.press("j")
                return "⏪ -10s"
            if action == "yt_fullscreen":
                if PAG_OK: pyautogui.press("f")
                return "⛶ Fullscreen"
            if action == "yt_mute":
                if PAG_OK: pyautogui.press("m")
                return "🔇 YT Mute"

            # ── Volume ───────────────────────────────────────────
            if action == "vol_up":
                if PAG_OK:
                    for _ in range(3): pyautogui.press("volumeup")
                return "🔊 Vol+"
            if action == "vol_down":
                if PAG_OK:
                    for _ in range(3): pyautogui.press("volumedown")
                return "🔉 Vol-"
            if action in ("vol_mute","mute"):
                if PAG_OK: pyautogui.press("volumemute")
                return "🔇 Mute"

            # ── System ───────────────────────────────────────────
            if action == "screenshot":
                if PAG_OK: pyautogui.hotkey("win","shift","s")
                return "📸 Screenshot"
            if action == "show_desktop":
                if PAG_OK: pyautogui.hotkey("win","d")
                return "🖥 Desktop"
            if action == "copy":
                if PAG_OK: pyautogui.hotkey("ctrl","c")
                return "📋 Copy"
            if action == "paste":
                if PAG_OK: pyautogui.hotkey("ctrl","v")
                return "📌 Paste"
            if action == "cut":
                if PAG_OK: pyautogui.hotkey("ctrl","x")
                return "✂ Cut"
            if action == "undo":
                if PAG_OK: pyautogui.hotkey("ctrl","z")
                return "↩ Undo"
            if action == "redo":
                if PAG_OK: pyautogui.hotkey("ctrl","y")
                return "↪ Redo"
            if action == "save":
                if PAG_OK: pyautogui.hotkey("ctrl","s")
                return "💾 Save"
            if action == "select_all":
                if PAG_OK: pyautogui.hotkey("ctrl","a")
                return "Select All"
            if action == "task_manager":
                if PAG_OK: pyautogui.hotkey("ctrl","shift","esc")
                return "📊 Task Manager"
            if action == "close_window":
                if PAG_OK: pyautogui.hotkey("alt","f4")
                return "✕ Closed"
            if action in ("win_minimize","minimize"):
                if PAG_OK: pyautogui.hotkey("win","down")
                return "▁ Minimized"
            if action in ("win_maximize","maximize"):
                if PAG_OK: pyautogui.hotkey("win","up")
                return "□ Maximized"
            if action == "win_restore":
                if PAG_OK: pyautogui.hotkey("win","up")
                return "⧉ Restored"
            if action == "alt_tab":
                if PAG_OK: pyautogui.hotkey("alt","tab")
                return "⇄ Alt+Tab"
            if action == "task_view":
                if PAG_OK: pyautogui.hotkey("win","tab")
                return "⊞ Task View"
            if action == "lock":
                if CTYPES_OK: ctypes.windll.user32.LockWorkStation()
                return "🔒 Locked"
            if action == "shutdown":
                subprocess.Popen(["shutdown","/s","/t","10"])
                return "⏻ Shutdown in 10s"
            if action == "scroll_up":
                if PAG_OK: pyautogui.scroll(5)
                return "↑ Scroll"
            if action == "scroll_down":
                if PAG_OK: pyautogui.scroll(-5)
                return "↓ Scroll"
            if action == "tab_key":
                if PAG_OK: pyautogui.press("tab")
                return "⇥ Tab"
            if action == "go_back":
                if PAG_OK: pyautogui.hotkey("alt","left")
                return "← Back"
            if action == "go_forward":
                if PAG_OK: pyautogui.hotkey("alt","right")
                return "→ Forward"
            if action == "new_tab":
                if PAG_OK: pyautogui.hotkey("ctrl","t")
                return "＋ New Tab"
            if action == "find":
                if PAG_OK: pyautogui.hotkey("ctrl","f")
                return "🔍 Find"
            if action == "new_file":
                if PAG_OK: pyautogui.hotkey("ctrl","n")
                return "📄 New"

            # Key aliases
            if action == "key_backspace":
                if PAG_OK: pyautogui.press("backspace")
                return "⌫ Del"
            if action == "key_enter":
                if PAG_OK: pyautogui.press("enter")
                return "↩ Enter"
            if action == "key_tab":
                if PAG_OK: pyautogui.press("tab")
                return "⇥ Tab"
            if action == "key_space":
                if PAG_OK: pyautogui.press("space")
                return "⎵ Space"

            # ── YouTube search bar focus ──────────────────────────
            if action == "yt_focus_search":
                if PAG_OK:
                    pyautogui.press("/")   # YouTube keyboard shortcut
                return "🔍 YT Search focused"
            if action == "yt_home":
                webbrowser.open("https://www.youtube.com")
                return "🏠 YT Home"
            if action == "yt_search":
                q = data.get("query","").strip()
                if q:
                    url = f"https://www.youtube.com/results?search_query={q.replace(' ','+')}"
                    webbrowser.open(url)
                    return f"🔍 YT: {q}"
                return "No query"

            # ── Google / browser search ───────────────────────────
            if action == "google_focus_search":
                if PAG_OK:
                    pyautogui.hotkey("ctrl","l")  # focus address bar
                return "🔍 Address bar focused"
            if action == "google_search":
                q = data.get("query","").strip()
                if q:
                    url = f"https://www.google.com/search?q={q.replace(' ','+')}"
                    webbrowser.open(url)
                    return f"🔍 Google: {q}"
                return "No query"

            # ── Open URL ─────────────────────────────────────────
            if action == "open_url":
                url = data.get("url","").strip()
                if url:
                    if not url.startswith("http"):
                        url = "https://" + url
                    webbrowser.open(url)
                    return f"🌐 {url[:40]}"
                return "No URL"

            # ── Window management by hwnd ─────────────────────────
            if action == "win_focus":
                hwnd = data.get("hwnd",0)
                if hwnd and self._focus_window(int(hwnd)):
                    return "✓ Window focused"
                return "⚠ Focus failed"
            if action == "win_close_hwnd":
                hwnd = data.get("hwnd",0)
                if hwnd and self._close_window_hwnd(int(hwnd)):
                    return "✕ Window closed"
                return "⚠ Close failed"
            if action == "win_maximize_hwnd":
                hwnd = data.get("hwnd",0)
                if hwnd and self._maximize_window_hwnd(int(hwnd)):
                    return "□ Maximized"
                return "⚠ Failed"

            # ── Open Apps ────────────────────────────────────────
            APP_CMDS = {
                "open_youtube":     lambda: (_exec_voice_action("youtube"), "📺 YouTube"),
                "open_chrome":      lambda: (subprocess.Popen(["start","chrome"],shell=True), "🌐 Chrome"),
                "open_notepad":     lambda: (subprocess.Popen(["notepad.exe"]), "📝 Notepad"),
                "open_excel":       lambda: (subprocess.Popen(["start","excel"],shell=True), "📊 Excel"),
                "open_word":        lambda: (subprocess.Popen(["start","winword"],shell=True), "📄 Word"),
                "open_powerpoint":  lambda: (subprocess.Popen(["start","powerpnt"],shell=True), "📑 PowerPoint"),
                "open_calculator":  lambda: (subprocess.Popen(["calc.exe"]), "🔢 Calculator"),
                "open_paint":       lambda: (subprocess.Popen(["mspaint.exe"]), "🎨 Paint"),
                "open_explorer":    lambda: (subprocess.Popen(["explorer.exe"]), "📁 Explorer"),
                "open_settings":    lambda: (subprocess.Popen(["start","ms-settings:"],shell=True), "⚙ Settings"),
            }
            if action in APP_CMDS:
                try:
                    _, msg = APP_CMDS[action]()
                    return f"🚀 {msg}"
                except Exception as e:
                    return f"⚠ {e}"

            return f"Unknown: {action}"

        except Exception as e:
            return f"Error: {e}"


class AirScrollUI:

    def __init__(self):
        self.root           = tk.Tk()
        self.logs           = deque(maxlen=300)
        self._running       = False
        self._stop_flag     = [False, "", None]
        self._voice         = None
        self._voice_on      = False
        self._cam_var       = tk.IntVar(value=0)
        self._phone_server  = None
        self._phone_on      = False
        self._setup()
        self._build()
        self._anim_si = 0
        self._anim()
        self.root.after(200, self._try_set_icon)
        self.root.after(600, self._poll_mic_state)
        self._mirror = MirrorPanel(self.root)
        self.root.after(800, self._start_phone_server)  # auto-start server

    def run(self):
        self.root.mainloop()

    def _build(self):
        """Build the entire UI — header + body."""
        self._build_header()
        self._build_body()

    def _setup(self):
        self.root.title("AirScroll")
        self.root.geometry("900x640")
        self.root.minsize(700, 500)
        self.root.configure(bg=BG)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"900x640+{(sw-900)//2}+{(sh-640)//2}")
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TScrollbar", background=CARD2, troughcolor=BG,
                    arrowcolor=MUTED2, bordercolor=BDR)

    def _try_set_icon(self):
        try:
            import io
            from PIL import Image, ImageDraw, ImageTk
            sz  = 64
            img = Image.new("RGBA", (sz,sz), (0,0,0,0))
            d   = ImageDraw.Draw(img)
            d.rounded_rectangle([2,2,sz-2,sz-2], radius=12, fill=(245,196,0,255))
            d.rounded_rectangle([8,8,sz-8,sz-8], radius=8,  fill=(10,10,25,255))
            d.text((sz//2-10, sz//2-10), "✋", fill=(245,196,0,255))
            buf = io.BytesIO()
            img.save(buf, format="PNG"); buf.seek(0)
            self._icon = ImageTk.PhotoImage(data=buf.read())
            self.root.iconphoto(True, self._icon)
        except Exception: pass
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "CIT.AirScroll.1.0")
        except Exception: pass

    # ── Build UI ─────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG2, height=72)
        hdr.pack(fill="x", padx=12, pady=(12,0))
        hdr.pack_propagate(False)

        # Logo
        logo = tk.Canvas(hdr, width=46, height=46, bg=BG2, highlightthickness=0)
        logo.pack(side="left", padx=(16,12), pady=13)
        logo.create_rectangle(0,0,46,46, fill=GOLD, outline="")
        logo.create_text(23,23, text="✋", font=("Segoe UI Emoji",22))

        # Title
        tf = tk.Frame(hdr, bg=BG2); tf.pack(side="left", pady=14)
        tk.Label(tf, text="AIRSCROLL", font=("Courier",17,"bold"),
                 fg=GOLD, bg=BG2).pack(anchor="w")
        tk.Label(tf, text="PALM GESTURE + VOICE  •  CIT JAGDALPUR",
                 font=("Courier",8), fg=MUTED2, bg=BG2).pack(anchor="w")

        # Right controls
        right = tk.Frame(hdr, bg=BG2)
        right.pack(side="right", padx=16, pady=12)

        # Status dot
        sf = tk.Frame(right, bg=CARD2,
                      highlightbackground=BDR, highlightthickness=1)
        sf.pack(side="right", padx=(8,0))
        self._dot = tk.Label(sf, text="●", font=("Courier",13),
                             fg=GREEN, bg=CARD2, padx=8, pady=5)
        self._dot.pack(side="left")
        self._status_lbl = tk.Label(sf, text="Ready",
                                     font=("Outfit",10,"bold"),
                                     fg=TEXT, bg=CARD2, pady=5, padx=4)
        self._status_lbl.pack(side="left")

        # Voice button
        self._voice_btn = tk.Button(
            right, text="🎤  VOICE OFF",
            font=("Courier",10,"bold"),
            bg=CARD2, fg=MUTED2, relief="flat",
            padx=12, pady=7, cursor="hand2",
            command=self._toggle_voice)
        self._voice_btn.pack(side="right", padx=(6,0))

        # Launch button
        self._launch_btn = tk.Button(
            right, text="▶  LAUNCH",
            font=("Courier",11,"bold"),
            bg=GREEN, fg=BG, relief="flat",
            padx=18, pady=7, cursor="hand2",
            command=self._toggle_engine)
        self._launch_btn.pack(side="right", padx=(6,0))

        # ── Voice ticker bar ─────────────────────────────────────
        # Persistent green bar below header showing real-time heard text
        ticker = tk.Frame(self.root, bg="#0a1a0a", height=32)
        ticker.pack(fill="x", padx=0, pady=0)
        ticker.pack_propagate(False)

        # Mic icon
        tk.Label(ticker, text=" 🎤",
                 font=("Segoe UI Emoji",11),
                 fg=GREEN, bg="#0a1a0a").pack(side="left", padx=(8,4))

        # Heard text label — updates in real time
        self._voice_ticker_var = tk.StringVar(value="Say a command...")
        self._voice_ticker_lbl = tk.Label(
            ticker,
            textvariable=self._voice_ticker_var,
            font=("Courier", 11, "bold"),
            fg=GREEN, bg="#0a1a0a",
            anchor="w")
        self._voice_ticker_lbl.pack(side="left", fill="x", expand=True)

        # Mic state badge (right side)
        self._mic_state_var = tk.StringVar(value="")
        self._mic_state_lbl = tk.Label(
            ticker,
            textvariable=self._mic_state_var,
            font=("Courier", 9, "bold"),
            fg=BG, bg=MUTED, padx=8, pady=4)
        self._mic_state_lbl.pack(side="right", padx=8)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=10)
        body.columnconfigure(0, weight=2)   # log
        body.columnconfigure(1, weight=1)   # voice ref
        body.rowconfigure(0, weight=0)
        body.rowconfigure(1, weight=1)

        self._build_gesture_cards(body)
        self._build_log(body)
        self._build_voice_ref(body)
        self._build_phone_panel(body)
        self._build_settings(body)

    def _build_gesture_cards(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.grid(row=0, column=0, columnspan=2, sticky="ew",
                   pady=(0,10))

        cards = [
            ("🖐←",  "PALM LEFT",   "← Arrow\nSwipe x2 = Hold ←",  ORANGE),
            ("🖐→",  "PALM RIGHT",  "→ Arrow\nSwipe x2 = Hold →",  BLUE),
            ("🖐↑",  "PALM UP",     "↑ Arrow\nSwipe x2 = Hold ↑",  GREEN),
            ("🖐↓",  "PALM DOWN",   "↓ Arrow\nSwipe x2 = Hold ↓",  CYAN),
            ("☝ TAP", "INDEX FINGER", "Point & hold 1.2s\n= Click",  GOLD),
            ("🤟SCR", "3-FINGER",   "Move up/down\n= Scroll",        PURPLE),
        ]
        frame.columnconfigure((0,1,2,3,4,5), weight=1)

        for i, (emoji, title, desc, color) in enumerate(cards):
            card = tk.Frame(frame, bg=CARD,
                            highlightbackground=color,
                            highlightthickness=2)
            card.grid(row=0, column=i, padx=5, sticky="nsew")

            tk.Frame(card, bg=color, height=3).pack(fill="x")
            body = tk.Frame(card, bg=CARD, padx=14, pady=12)
            body.pack(fill="both")

            tk.Label(body, text=emoji, font=("Segoe UI Emoji",28),
                     bg=CARD, fg=color).pack()
            tk.Label(body, text=title, font=("Courier",9,"bold"),
                     fg=color, bg=CARD).pack(pady=(4,2))
            tk.Label(body, text=desc, font=("Outfit",9),
                     fg=MUTED2, bg=CARD, justify="center").pack()

    def _build_log(self, parent):
        lf = tk.Frame(parent, bg=BG)
        lf.grid(row=1, column=0, sticky="nsew", padx=(0,6))

        top = tk.Frame(lf, bg=BG)
        top.pack(fill="x", pady=(0,4))
        tk.Label(top, text="ACTIVITY LOG", font=("Courier",9),
                 fg=MUTED, bg=BG).pack(side="left")
        tk.Button(top, text="Clear", font=("Outfit",8),
                  bg=CARD2, fg=RED, relief="flat",
                  padx=8, pady=2, cursor="hand2",
                  command=self._clear_log).pack(side="right")

        self._log_text = tk.Text(
            lf, bg=CARD, fg=TEXT,
            font=("Courier",9), relief="flat",
            state="disabled", wrap="word",
            highlightbackground=BDR, highlightthickness=1)
        sb = ttk.Scrollbar(lf, orient="vertical",
                           command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True)

        self._log("AirScroll ready — click LAUNCH to start")

    def _build_voice_ref(self, parent):
        vf = tk.Frame(parent, bg=BG)
        vf.grid(row=1, column=1, sticky="nsew", padx=(6,0))

        tk.Label(vf, text="VOICE COMMANDS", font=("Courier",9),
                 fg=MUTED, bg=BG, pady=(0)).pack(anchor="w", pady=(0,4))

        cvs = tk.Canvas(vf, bg=BG, highlightthickness=0)
        sb  = ttk.Scrollbar(vf, orient="vertical", command=cvs.yview)
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cvs.pack(fill="both", expand=True)
        inner = tk.Frame(cvs, bg=BG)
        cw = cvs.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.bind("<Configure>",
                 lambda e: cvs.itemconfig(cw, width=e.width))

        sections = [
            (GOLD, [
                ("open youtube",    "YouTube"),
                ("open chrome",     "Chrome"),
                ("open excel",      "Excel"),
                ("open notepad",    "Notepad"),
                ("screenshot",      "Screenshot"),
                ("lock screen",     "Lock PC"),
            ]),
            (CYAN, [
                ("volume up/down",  "Volume"),
                ("mute",            "Toggle mute"),
                ("play / pause",    "Media"),
                ("next track",      "Next song"),
            ]),
            (GREEN, [
                ("press up/down",   "Arrow keys"),
                ("press left/right","Arrow keys"),
                ("scroll up/down",  "Scroll"),
                ("go back",         "Alt+Left"),
            ]),
            (PURPLE, [
                ("copy / paste",    "Clipboard"),
                ("undo / save",     "Edit"),
                ("search <query>",  "Google"),
                ("go to <url>",     "Open URL"),
            ]),
        ]
        for color, cmds in sections:
            for cmd, act in cmds:
                row = tk.Frame(inner, bg=CARD,
                               highlightbackground=BDR, highlightthickness=1)
                row.pack(fill="x", pady=1)
                tk.Label(row, text=f'"{cmd}"',
                         font=("Courier",8,"bold"), fg=color,
                         bg=CARD2, padx=6, pady=3).pack(side="left")
                tk.Label(row, text=f"  {act}",
                         font=("Outfit",9), fg=MUTED2,
                         bg=CARD).pack(side="left")

    def _build_settings(self, parent):
        sf = tk.Frame(parent, bg=BG2,
                      highlightbackground=BDR, highlightthickness=1)
        sf.grid(row=2, column=0, columnspan=2,
                sticky="ew", pady=(10,0))

        tk.Label(sf, text="  Camera:",
                 font=("Outfit",9), fg=MUTED2, bg=BG2,
                 pady=8).pack(side="left")
        tk.Spinbox(sf, from_=0, to=5, width=3,
                   textvariable=self._cam_var,
                   font=("Outfit",10), bg=CARD2, fg=TEXT,
                   buttonbackground=CARD2, relief="flat").pack(side="left", padx=(4,16))

        tk.Label(sf, text="🧠 Claude API Key (for Hinglish/NLU):",
                 font=("Outfit",9), fg=CYAN, bg=BG2).pack(side="left")
        self._api_key_var = tk.StringVar(value=self._load_api_key())
        api_entry = tk.Entry(sf, textvariable=self._api_key_var,
                             font=("Courier",9), bg=CARD2, fg=TEXT,
                             insertbackground=TEXT, relief="flat",
                             width=36, show="*")
        api_entry.pack(side="left", padx=(6,4), ipady=4)
        # Toggle show/hide key
        def toggle_show():
            api_entry.config(show="" if api_entry.cget("show")=="*" else "*")
        tk.Button(sf, text="👁", font=("Courier",9),
                  bg=CARD2, fg=MUTED2, relief="flat",
                  padx=4, pady=3, cursor="hand2",
                  command=toggle_show).pack(side="left")
        tk.Button(sf, text="💾", font=("Courier",9),
                  bg=GOLD, fg=BG, relief="flat",
                  padx=6, pady=3, cursor="hand2",
                  command=self._save_api_key).pack(side="left", padx=(4,0))

        tk.Label(sf, text="  Vaibhav • Aashu • Agash  |  CIT Jagdalpur  ",
                 font=("Courier",8), fg=MUTED, bg=BG2,
                 pady=8).pack(side="right")

    # ── API Key ───────────────────────────────────────────────────
    def _load_api_key(self):
        try:
            with open("airscroll_api_key.txt") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _save_api_key(self):
        key = self._api_key_var.get().strip()
        try:
            with open("airscroll_api_key.txt","w") as f:
                f.write(key)
            self._log("🧠 API key saved")
            self._toast("✓ API key saved")
            # Push key to voice engine
            if self._voice:
                self._voice._api_key = key
        except Exception as e:
            self._toast(f"⚠ {e}", False)

    # ── Engine toggle ─────────────────────────────────────────────
    def _toggle_mirror(self):
        if self._mirror and self._mirror.is_visible():
            self._mirror.hide()
            try: self._mirror_btn.config(text="🖥  MIRROR OFF", bg=CARD2, fg=MUTED2)
            except Exception: pass
            self._log("🖥 Mirror hidden")
        else:
            if self._mirror: self._mirror.show()
            try: self._mirror_btn.config(text="🖥  MIRROR ON", bg=CYAN, fg=BG)
            except Exception: pass
            self._log("🖥 Mirror shown")

    def _build_phone_panel(self, parent):
        """QR code + phone info panel embedded in main UI."""
        pf = tk.Frame(parent, bg=BG2,
                      highlightbackground=BDR, highlightthickness=1)
        pf.grid(row=1, column=2, sticky="nsew", padx=(6,0))
        parent.columnconfigure(2, weight=0)

        tk.Label(pf, text="📱 PHONE REMOTE",
                 font=("Courier",9,"bold"), fg=GOLD, bg=BG2,
                 pady=6).pack(anchor="w", padx=8)

        self._phone_url_var = tk.StringVar(value="Starting server...")
        tk.Label(pf, textvariable=self._phone_url_var,
                 font=("Courier",8), fg=CYAN, bg=BG2,
                 wraplength=160).pack(padx=8, pady=2)

        # QR code canvas
        self._qr_canvas = tk.Canvas(pf, width=160, height=160,
                                     bg=BG2, highlightthickness=0)
        self._qr_canvas.pack(padx=8, pady=4)
        self._qr_canvas.create_text(80, 80, text="⏳ Starting...",
                                     fill=MUTED2, font=("Courier",9))

        tk.Label(pf, text="Scan QR with phone  (same WiFi)",
                 font=("Outfit",8), fg=MUTED2, bg=BG2,
                 justify="center").pack(pady=2)

        # Instructions
        steps = [
            "1. Connect phone to same WiFi",
            "2. Scan QR code or open URL",
            "3. Control PC from browser",
        ]
        for s in steps:
            tk.Label(pf, text=s, font=("Outfit",8),
                     fg=MUTED, bg=BG2, anchor="w").pack(anchor="w", padx=8)

        # Status
        self._phone_status_var = tk.StringVar(value="● Offline")
        self._phone_status_lbl = tk.Label(
            pf, textvariable=self._phone_status_var,
            font=("Courier",9,"bold"), fg=RED, bg=BG2)
        self._phone_status_lbl.pack(pady=4)

    def _start_phone_server(self):
        """Auto-start phone server on launch."""
        if not FLASK_OK:
            self._log("📱 Install flask: pip install flask")
            try:
                self._phone_url_var.set("pip install flask")
            except Exception: pass
            return
        self._phone_server = PhoneServer(ui_ref=self, port=5000)
        ok = self._phone_server.start()
        if ok:
            url = self._phone_server.get_url()
            self._phone_on = True
            try:
                self._phone_btn.config(text="📱  PHONE ON", bg=GREEN, fg=BG)
                self._phone_url_var.set(url)
                self._phone_status_var.set("● Online")
                self._phone_status_lbl.config(fg=GREEN)
            except Exception: pass
            self._log(f"📱 Phone server ON: {url}")
            self._draw_qr(url)
        else:
            self._log("📱 Flask not installed — pip install flask")

    def _toggle_phone_server(self):
        if self._phone_on and self._phone_server:
            self._phone_server.stop()
            self._phone_on = False
            self._phone_btn.config(text="📱  PHONE OFF", bg=CARD2, fg=MUTED2)
            try:
                self._phone_status_var.set("● Offline")
                self._phone_status_lbl.config(fg=RED)
            except Exception: pass
            self._log("📱 Phone server stopped")
        else:
            self._start_phone_server()

    def _draw_qr(self, url):
        """Draw QR code on canvas using qrcode library or ASCII fallback."""
        def _do():
            try:
                if QR_OK:
                    import qrcode, io
                    from PIL import Image, ImageTk
                    qr = qrcode.QRCode(box_size=4, border=2)
                    qr.add_data(url)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    img = img.resize((160,160), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    def _apply(p=photo):
                        try:
                            self._qr_canvas.delete("all")
                            self._qr_canvas.create_image(0,0,anchor="nw",image=p)
                            self._qr_canvas._photo = p  # keep ref
                        except Exception: pass
                    self.root.after(0, _apply)
                else:
                    # Text fallback — show URL prominently
                    def _text():
                        try:
                            self._qr_canvas.delete("all")
                            self._qr_canvas.create_rectangle(0,0,160,160,
                                fill="#ffffff", outline="#F5C400", width=2)
                            self._qr_canvas.create_text(80,60,
                                text="Open on phone:", fill="#07071A",
                                font=("Courier",8,"bold"))
                            self._qr_canvas.create_text(80,90,
                                text=url.replace("http://",""),
                                fill="#07071A", font=("Courier",9,"bold"),
                                width=140, justify="center")
                            self._qr_canvas.create_text(80,130,
                                text="pip install qrcode  for QR",
                                fill="#888", font=("Courier",7),
                                justify="center")
                        except Exception: pass
                    self.root.after(0, _text)
            except Exception as e:
                self._log(f"QR error: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def _show_mirror(self):
        if self._mirror:
            self._mirror.show()
            try: self._mirror_btn.config(text="🖥  MIRROR ON", bg=CYAN, fg=BG)
            except Exception: pass

    def _hide_mirror(self):
        if self._mirror:
            self._mirror.hide()
            try: self._mirror_btn.config(text="🖥  MIRROR OFF", bg=CARD2, fg=MUTED2)
            except Exception: pass

    def _toggle_engine(self):
        if self._running:
            # STOP — engine off, voice off, mirror off
            self._stop_flag[0] = True
            self._running      = False
            self._launch_btn.config(text="▶  LAUNCH", bg=GREEN)
            self._status_lbl.config(text="Ready")
            self._log("Engine stopped")
            self._toast("⏹ Stopped")
            if self._voice_on:
                self.root.after(200, self._toggle_voice)
            self.root.after(400, self._hide_mirror)   # hide mirror on stop
        else:
            # LAUNCH — engine on, voice on, mirror on
            if not CV2_OK:
                messagebox.showerror("Missing",
                    "OpenCV / MediaPipe not installed.\n\n"
                    "pip install opencv-python mediapipe")
                return
            cam = self._cam_var.get()
            self._stop_flag = [False, "", None]
            self._running   = True
            self._launch_btn.config(text="⏹  STOP", bg=RED)
            self._status_lbl.config(text="Running")
            self._log(f"Engine started  (camera {cam})")
            self._toast("▶ Launched!")
            if not self._voice_on:
                self.root.after(500, self._toggle_voice)
            self.root.after(300, self._show_mirror)   # show mirror on launch

            def _run():
                try:
                    run_engine(cam, self._log, self._stop_flag)
                except Exception as e:
                    self._log(f"⚠ {e}")
                finally:
                    self._running = False
                    self.root.after(0, lambda: (
                        self._launch_btn.config(text="▶  LAUNCH", bg=GREEN),
                        self._status_lbl.config(text="Ready"),
                    ))
                    if self._voice_on:
                        self.root.after(300, self._toggle_voice)
                    self.root.after(500, self._hide_mirror)
            threading.Thread(target=_run, daemon=True).start()

    def _toggle_voice(self):
        if self._voice_on:
            if self._voice: self._voice.stop(); self._voice = None
            self._voice_on = False
            self._voice_btn.config(text="🎤  VOICE OFF", bg=CARD2, fg=MUTED2)
            self._log("🎤 Voice off")
        else:
            if not SR_OK:
                messagebox.showerror("Missing",
                    "SpeechRecognition not installed.\n\n"
                    "pip install SpeechRecognition\n"
                    "pip install pyaudio\n\n"
                    "If pyaudio fails:\n"
                    "pip install pipwin\n"
                    "pipwin install pyaudio")
                return

            def v_log(msg):
                self._log(msg)
                if self._running and len(self._stop_flag) > 1:
                    self._stop_flag[1] = msg
                # Handle mirror panel signals from voice
                if "show_mirror_signal" in msg:
                    self.root.after(0, lambda: self._show_mirror())
                elif "hide_mirror_signal" in msg:
                    self.root.after(0, lambda: self._hide_mirror())
                # Update voice ticker bar in real time
                self.root.after(0, lambda m=msg: self._update_voice_ticker(m))

            self._voice = VoiceEngine(log_fn=v_log,
                                       api_key=self._api_key_var.get().strip())

            self._voice._mirror = getattr(self,"_mirror",None)
            if self._voice.start():
                self._voice_on = True
                self._stop_flag[2] = self._voice
                self._voice_btn.config(text="🎤  VOICE ON", bg=GREEN, fg=BG)
                self._log("🎤 Voice on — speak a command!")
                self._toast("🎤 Voice ON!")
            else:
                self._voice = None
                self._toast("⚠ Mic not available", False)

    # ── Log ──────────────────────────────────────────────────────
    def _update_voice_ticker(self, msg):
        """Update the green voice ticker bar with latest heard/action text."""
        if not hasattr(self, '_voice_ticker_var'): return
        try:
            # Only show meaningful messages — filter out internal signals
            skip = ("show_mirror_signal","hide_mirror_signal","🧠","⚠","[DWELL]")
            if any(s in msg for s in skip): return

            # Heard text — strip emoji prefix for clean display
            display = msg.strip()
            if display.startswith("🎤"):
                # Raw heard text — show prominently
                heard = display.replace("🎤","").replace('"','').strip()
                self._voice_ticker_var.set(f'  "{heard}"')
                self._voice_ticker_lbl.config(fg=GREEN)
                # Flash green bg briefly
                self._voice_ticker_lbl.config(bg="#0d2a0d")
                self.root.after(600, lambda: self._voice_ticker_lbl.config(bg="#0a1a0a"))
            else:
                # Action/result message
                self._voice_ticker_var.set(f"  {display[:80]}")
                self._voice_ticker_lbl.config(fg=CYAN)

            # Auto-clear after 4 seconds
            self.root.after(4000, self._clear_ticker)
        except Exception: pass

    def _clear_ticker(self):
        try:
            self._voice_ticker_var.set("Say a command...")
            self._voice_ticker_lbl.config(fg=MUTED, bg="#0a1a0a")
        except Exception: pass

    def _poll_mic_state(self):
        """Update mic state badge every 500ms."""
        try:
            if hasattr(self, '_mic_state_var') and self._voice:
                paused = getattr(self._voice, '_mic_paused', False)
                if paused:
                    self._mic_state_var.set(" 🔇 MIC PAUSED ")
                    self._mic_state_lbl.config(bg=RED, fg=BG)
                elif self._voice_on:
                    self._mic_state_var.set(" 🎤 LISTENING ")
                    self._mic_state_lbl.config(bg=GREEN, fg=BG)
                else:
                    self._mic_state_var.set(" OFF ")
                    self._mic_state_lbl.config(bg=MUTED, fg=BG)
            else:
                if hasattr(self, '_mic_state_var'):
                    self._mic_state_var.set("")
        except Exception: pass
        self.root.after(500, self._poll_mic_state)

    def _log(self, msg):
        try:
            ts   = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}]  {msg}\n"
            self.logs.append(line)
            self.root.after(0, lambda: self._log_insert(line))
        except Exception: pass

    def _log_insert(self, line):
        try:
            self._log_text.configure(state="normal")
            self._log_text.insert("end", line)
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        except Exception: pass

    def _clear_log(self):
        try:
            self._log_text.configure(state="normal")
            self._log_text.delete("1.0","end")
            self._log_text.configure(state="disabled")
            self.logs.clear()
        except Exception: pass

    # ── Toast notification ────────────────────────────────────────
    def _toast(self, msg, ok=True):
        try:
            t = tk.Toplevel(self.root)
            t.overrideredirect(True)
            t.attributes("-topmost", True)
            bg = GREEN if ok else RED
            t.configure(bg=bg)
            txt = str(msg)[:70]
            tk.Label(t, text=f"  {txt}  ",
                     font=("Outfit",11,"bold"),
                     fg=BG, bg=bg, pady=9, padx=8).pack()
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            t.update_idletasks()
            t.geometry(f"+{sw-t.winfo_width()-18}+{sh-80}")
            t.after(2500, lambda: self._safe_destroy(t))
        except Exception: pass

    def _safe_destroy(self, w):
        try: w.destroy()
        except Exception: pass

    # ── Status dot animation ──────────────────────────────────────
    def _anim(self):
        colours = [GREEN, "#00CC66", "#009944"]
        def tick():
            try:
                self._dot.config(fg=colours[self._anim_si % 3])
                self._anim_si += 1
                self.root.after(700, tick)
            except Exception: pass
        tick()



# ══════════════════════════════════════════════════════════════════
#  GLOBAL ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════
def _global_exc(exc_type, exc_value, exc_tb):
    import traceback
    print("[AirScroll] Unhandled:", "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

def _thread_exc(args):
    import traceback
    print("[AirScroll thread]", "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

sys.excepthook         = _global_exc
threading.excepthook   = _thread_exc


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 52)
    print("  AirScroll  |  CGIT Jagdalpur")
    print("  Vaibhav Kaushik  •  Aashu Dewangan  •  Agash Kumar")
    print("=" * 52)
    try:
        AirScrollUI().run()
    except KeyboardInterrupt:
        print("\nBye!")
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            import tkinter as _tk
            r = _tk.Tk(); r.withdraw()
            from tkinter import messagebox as _mb
            _mb.showerror("AirScroll Error", str(e))
            r.destroy()
        except Exception: pass