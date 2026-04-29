#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ ADIMMA-KANN — Voice-First Telegram AI Assistant                             ║
║ Sharp. Sarcastic. Kerala-flavoured. Serving "sir" only.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  Flask webhook server (Gunicorn on Render.com)
  Groq (LLM: llama-3.3-70b-versatile | Vision: llama-4-scout | STT: Whisper)
  edge-tts → voice replies (Malayalam: ml-IN-SobhanaNeural, English: en-GB-RyanNeural)
  Pillow → image description preprocessing
  PyPDF2 → PDF text extraction
  langdetect → language detection
  OpenWeatherMap → weather queries (free tier)

Author : Built for "sir" 😏

FIXES vs original:
  - Webhook route is now sync (Flask doesn't natively support async routes).
  - Async edge-tts calls are wrapped in a fresh event loop (unchanged — already correct).
  - Added /setwebhook and /webhookinfo helper routes for easier Render deployment.
  - RENDER_EXTERNAL_URL auto-registers webhook on startup (unchanged).
  - All original features preserved: Groq LLM/Vision/STT, TTS, PDF, Weather,
    Malayalam/Manglish detection, sleep/wake, access control, conversation history.
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import re
import json
import time
import logging
import hashlib
import asyncio
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Optional, List, Dict, Tuple

# ── Third-Party ───────────────────────────────────────────────────────────────
import requests
import edge_tts
from flask import Flask, request, jsonify
from groq import Groq
from PIL import Image
import PyPDF2
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Make langdetect deterministic
DetectorFactory.seed = 42

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AdimmaKann")

# ── Environment Variables ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OWNER_ID = int(os.getenv("OWNER_ID", "733340342"))
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Groq API key pool — rotated round-robin for reliability
_GROQ_KEYS_RAW = [
    os.getenv("GROQ_API_KEY", ""),
    os.getenv("GROQ_API_KEY1", ""),
    os.getenv("GROQ_API_KEY2", ""),
    os.getenv("GROQ_API_KEY3", ""),
]
GROQ_API_KEYS: List[str] = [k for k in _GROQ_KEYS_RAW if k.strip()]
if not GROQ_API_KEYS:
    logger.critical("No Groq API keys found! Set GROQ_API_KEY in environment.")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Allowed groups / users (comma-separated IDs in env)
_raw_groups = os.getenv("TELEGRAM_ALLOWED_GROUPS", "")
_raw_users  = os.getenv("TELEGRAM_ALLOWED_USERS", str(OWNER_ID))
ALLOWED_GROUPS: List[int] = [int(x) for x in _raw_groups.split(",") if x.strip()]
ALLOWED_USERS:  List[int] = [int(x) for x in _raw_users.split(",")  if x.strip()]

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", "10000"))

# ── Model IDs ─────────────────────────────────────────────────────────────────
LLM_MODEL    = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
STT_MODEL    = "whisper-large-v3"

# ── TTS Voices ────────────────────────────────────────────────────────────────
TTS_VOICE_ENGLISH   = "en-GB-RyanNeural"      # British, slightly posh = fits the wit
TTS_VOICE_MALAYALAM = "ml-IN-SobhanaNeural"   # Female ML — crisp, clear
TTS_VOICE_MANGLISH  = "en-IN-PrabhatNeural"   # Indian English for Manglish mix
TTS_VOICE_FALLBACK  = "en-US-GuyNeural"

TTS_CACHE_DIR = Path("tts_cache")
TTS_CACHE_DIR.mkdir(exist_ok=True)

# ── Character Prompt ──────────────────────────────────────────────────────────
CHARACTER_FILE = Path("character.txt")

def load_character() -> str:
    """Load personality system prompt from character.txt."""
    if CHARACTER_FILE.exists():
        text = CHARACTER_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    logger.warning("character.txt not found or empty — using fallback persona.")
    return (
        "You are Adimma Kann, a witty, sarcastic, and slightly roasting AI assistant "
        "from Kerala, India. You serve only 'sir' — your one and only human. "
        "You speak like a clever Malayali friend: sharp, funny, warm but never boring. "
        "Keep responses concise and entertaining."
    )

SYSTEM_PROMPT = load_character()

# ── Conversation History ──────────────────────────────────────────────────────
MAX_HISTORY = 20
# chat_id → deque of {"role": "user"/"assistant", "content": "..."}
conversation_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

# ── Sleep/Wake State ──────────────────────────────────────────────────────────
# chat_id → True = sleeping (not responding)
sleeping_chats: Dict[int, bool] = defaultdict(lambda: False)

SLEEP_TRIGGERS = {"bye", "standby", "stop listening", "sleep", "good night", "goodnight", "stop"}
WAKE_TRIGGERS  = {"hi", "hello", "wake up", "adimma", "hey", "start", "yo"}

# ── Groq Client Pool ──────────────────────────────────────────────────────────
_groq_key_index = 0

def get_groq_client() -> Groq:
    """Return a Groq client, rotating through available API keys."""
    global _groq_key_index
    if not GROQ_API_KEYS:
        raise RuntimeError("No Groq API keys configured.")
    key = GROQ_API_KEYS[_groq_key_index % len(GROQ_API_KEYS)]
    _groq_key_index = (_groq_key_index + 1) % len(GROQ_API_KEYS)
    return Groq(api_key=key)

# ── Language Detection ────────────────────────────────────────────────────────
MALAYALAM_UNICODE_RANGE = range(0x0D00, 0x0D80)

# Common Manglish indicators — Romanised Malayalam words
MANGLISH_KEYWORDS = {
    "njan", "ningal", "avan", "aval", "itha", "ithu", "entha", "enthaa",
    "alle", "alla", "aayi", "aano", "okke", "sheriyaa", "sheriya",
    "cheyyum", "cheyyunnu", "paranjhu", "parayum", "kittum", "kittyo",
    "venam", "venda", "mone", "mole", "machane", "daaa", "daa", "di",
    "evidam", "ethra", "evidunnu", "poyi", "poda", "podi", "enthaa",
    "aadhyam", "pinne", "sheri", "adipoli", "pwoli",
    "nannayit", "nannayirunnu", "arike", "arikil", "kashtu", "kshtam",
    "enthelum", "enikku", "thante", "eppol", "eppo", "enga",
    "engane", "enganeyaa", "chetta", "chettan", "chechhi", "ammachi",
    "acha", "amma", "appan", "appupan", "vallyamma", "vallyappan",
    "mollee", "monee", "machaan", "machaney", "frnd", "ikka",
}

def detect_language(text: str) -> str:
    """
    Detect language of text.
    Returns: 'malayalam', 'manglish', or 'english'

    Strategy:
      1. If text contains Malayalam Unicode characters → 'malayalam'
      2. If text contains Manglish keywords → 'manglish'
      3. Use langdetect; if 'ml' → 'malayalam', else 'english'
    """
    if not text or not text.strip():
        return "english"

    # Step 1: Unicode Malayalam script check
    malayalam_chars = sum(1 for ch in text if ord(ch) in MALAYALAM_UNICODE_RANGE)
    if malayalam_chars > 0:
        return "malayalam"

    # Step 2: Manglish keyword check (case-insensitive)
    words_lower   = set(re.findall(r"\b\w+\b", text.lower()))
    manglish_hits = words_lower & MANGLISH_KEYWORDS
    total_words   = len(words_lower) or 1
    if manglish_hits and (len(manglish_hits) / total_words) >= 0.10:
        return "manglish"

    # Step 3: langdetect
    try:
        lang_code = detect(text)
        if lang_code == "ml":
            return "malayalam"
        # Sometimes Manglish gets detected as other Indian languages
        if lang_code in ("ta", "te", "kn", "hi", "bn", "mr", "gu"):
            if manglish_hits:
                return "manglish"
    except LangDetectException:
        pass

    return "english"

# ── TTS ───────────────────────────────────────────────────────────────────────
def get_tts_voice(language: str) -> str:
    """Select the best TTS voice for detected language."""
    voice_map = {
        "malayalam": TTS_VOICE_MALAYALAM,
        "manglish":  TTS_VOICE_MANGLISH,
        "english":   TTS_VOICE_ENGLISH,
    }
    return voice_map.get(language, TTS_VOICE_FALLBACK)

def tts_cache_path(text: str, voice: str) -> Path:
    """Generate a deterministic cache file path using MD5."""
    key = hashlib.md5(f"{voice}::{text}".encode("utf-8")).hexdigest()
    return TTS_CACHE_DIR / f"{key}.mp3"

async def _async_tts(text: str, voice: str, out_path: Path) -> None:
    """Async edge-tts synthesis."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))

def synthesize_speech(text: str, language: str = "english") -> Optional[Path]:
    """
    Convert text to speech using edge-tts.
    Returns path to mp3 file, or None on failure.
    Uses MD5-based caching to avoid re-generating identical audio.
    """
    voice = get_tts_voice(language)

    # Trim very long texts to avoid edge-tts timeouts
    if len(text) > 1000:
        text = text[:997] + "..."

    cache_file = tts_cache_path(text, voice)
    if cache_file.exists():
        logger.debug(f"TTS cache hit: {cache_file.name}")
        return cache_file

    try:
        # Run async TTS in a new event loop (we're inside sync Flask context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_async_tts(text, voice, cache_file))
        finally:
            loop.close()

        logger.info(f"TTS synthesised [{language}|{voice}]: {text[:60]}…")
        return cache_file
    except Exception as e:
        logger.error(f"TTS failed ({voice}): {e}")
        # Try fallback voice
        if voice != TTS_VOICE_FALLBACK:
            try:
                cache_file_fb = tts_cache_path(text, TTS_VOICE_FALLBACK)
                loop2 = asyncio.new_event_loop()
                asyncio.set_event_loop(loop2)
                try:
                    loop2.run_until_complete(_async_tts(text, TTS_VOICE_FALLBACK, cache_file_fb))
                finally:
                    loop2.close()
                return cache_file_fb
            except Exception as e2:
                logger.error(f"TTS fallback also failed: {e2}")
        return None

# ── Groq STT ──────────────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str, hint_language: Optional[str] = None) -> Tuple[str, str]:
    """
    Transcribe audio file using Groq Whisper.
    Returns (transcript_text, detected_language).
    Tries Malayalam first, then falls back to auto-detect.
    """
    lang_hints = []
    if hint_language == "malayalam":
        lang_hints = ["ml", None]
    else:
        lang_hints = [None, "ml", "en"]

    last_error = None
    for lang in lang_hints:
        try:
            client = get_groq_client()
            with open(audio_path, "rb") as f:
                kwargs = {
                    "file": (os.path.basename(audio_path), f.read()),
                    "model": STT_MODEL,
                    "response_format": "verbose_json",
                    "temperature": 0.0,
                }
                if lang:
                    kwargs["language"] = lang

                transcription = client.audio.transcriptions.create(**kwargs)

            text     = transcription.text.strip()
            detected = getattr(transcription, "language", "en") or "en"
            logger.info(f"STT [{lang or 'auto'}] → '{text[:80]}' (detected: {detected})")
            return text, detected
        except Exception as e:
            last_error = e
            logger.warning(f"STT attempt (lang={lang}) failed: {e}")
            time.sleep(0.5)

    raise RuntimeError(f"All STT attempts failed. Last error: {last_error}")

# ── Vision / Image Analysis ───────────────────────────────────────────────────
def describe_image(image_path: str) -> str:
    """
    Use Groq vision model to describe an image.
    Returns a textual description to inject into the LLM prompt.
    """
    try:
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        # Detect mime type
        img  = Image.open(image_path)
        fmt  = (img.format or "JPEG").upper()
        mime = f"image/{fmt.lower()}"
        if mime == "image/jpg":
            mime = "image/jpeg"

        client   = get_groq_client()
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail. Include objects, people, "
                                "text visible, context, mood, and anything notable. "
                                "Be thorough but concise."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        description = response.choices[0].message.content.strip()
        logger.info(f"Image described: {description[:80]}…")
        return description
    except Exception as e:
        logger.error(f"Image description failed: {e}")
        return "[Image received but could not be analysed at this moment.]"

# ── PDF Processing ────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str, max_chars: int = 4000) -> str:
    """Extract text from PDF. Truncates at max_chars for LLM context safety."""
    try:
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader    = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                chunk = page.extract_text() or ""
                text_parts.append(f"[Page {i+1}]\n{chunk}")
                if sum(len(t) for t in text_parts) > max_chars:
                    text_parts.append(f"\n... (truncated after page {i+1} of {num_pages})")
                    break

        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            return "[PDF received but no readable text could be extracted.]"
        logger.info(f"PDF extracted: {len(full_text)} chars from {num_pages} pages")
        return full_text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return "[PDF received but extraction failed — might be scanned/image-based.]"

# ── Weather Tool ──────────────────────────────────────────────────────────────
def get_weather(city: str) -> str:
    """Fetch current weather for a city using OpenWeatherMap free API."""
    if not OPENWEATHER_API_KEY:
        return "[Weather API key not configured, sir. Add OPENWEATHER_API_KEY to env.]"
    try:
        url    = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q":     city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        temp        = data["main"]["temp"]
        feels_like  = data["main"]["feels_like"]
        humidity    = data["main"]["humidity"]
        description = data["weather"][0]["description"].capitalize()
        wind_speed  = data["wind"]["speed"]
        city_name   = data["name"]
        country     = data["sys"]["country"]

        return (
            f"🌤 Weather in {city_name}, {country}:\n"
            f"  Condition : {description}\n"
            f"  Temp      : {temp}°C (feels like {feels_like}°C)\n"
            f"  Humidity  : {humidity}%\n"
            f"  Wind      : {wind_speed} m/s"
        )
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return f"[City '{city}' not found. Double-check the spelling, sir.]"
        return f"[Weather fetch failed: {e}]"
    except Exception as e:
        logger.error(f"Weather error: {e}")
        return f"[Weather service hiccup: {e}]"

def maybe_handle_weather(text: str) -> Optional[str]:
    """Check if user is asking about weather; if so, return weather info string."""
    pattern = re.search(
        r"\b(?:weather|temperature|temp|climate|rain|sunny|humid)\b.{0,20}?(?:in|at|for)?\s+([A-Za-z\s]{2,30})",
        text,
        re.IGNORECASE,
    )
    if pattern:
        city = pattern.group(1).strip().rstrip("?.,!")
        if city:
            return get_weather(city)
    return None

# ── LLM Chat ──────────────────────────────────────────────────────────────────
def chat_with_groq(
    chat_id: int,
    user_message: str,
    language: str = "english",
    extra_context: Optional[str] = None,
) -> str:
    """
    Send message to Groq LLM with conversation history.
    Returns assistant's reply text.
    """
    history = conversation_history[chat_id]

    # Build language instruction
    lang_instructions = {
        "malayalam": (
            "IMPORTANT: The user wrote in Malayalam. You MUST reply entirely in Malayalam "
            "(use Malayalam Unicode script). Do not switch to English."
        ),
        "manglish": (
            "IMPORTANT: The user wrote in Manglish (Romanised Malayalam mixed with English). "
            "Reply in natural Manglish — Malayalam words written in English letters, "
            "casually mixed with English. Example style: 'Sheri, njan check cheyyam sir!'"
        ),
        "english": "",
    }
    lang_note = lang_instructions.get(language, "")

    # Weather check
    weather_info = maybe_handle_weather(user_message)

    # Build system message
    system_content = SYSTEM_PROMPT
    if lang_note:
        system_content += f"\n\n{lang_note}"
    if weather_info:
        system_content += f"\n\n[WEATHER DATA — use this in your reply]:\n{weather_info}"
    if extra_context:
        system_content += f"\n\n[CONTEXT — user shared media]:\n{extra_context}"

    messages = [{"role": "system", "content": system_content}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": user_message})

    try:
        client   = get_groq_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.85,
            top_p=0.95,
        )
        reply = response.choices[0].message.content.strip()

        # Update history
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": reply})

        logger.info(f"LLM reply [{language}] for chat {chat_id}: {reply[:80]}…")
        return reply

    except Exception as e:
        logger.error(f"Groq LLM error: {e}\n{traceback.format_exc()}")
        # Rotate key and retry once
        try:
            client   = get_groq_client()
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.85,
            )
            reply = response.choices[0].message.content.strip()
            history.append({"role": "user",      "content": user_message})
            history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e2:
            logger.error(f"Groq retry also failed: {e2}")
            return (
                "Ayyo, something went sideways on my end, sir. "
                "Even I have bad days apparently. Try again in a bit?"
            )

# ── Telegram API Helpers ──────────────────────────────────────────────────────
def tg(method: str, **kwargs) -> dict:
    """Make a Telegram Bot API call."""
    url = f"{TELEGRAM_API_BASE}/{method}"
    try:
        resp = requests.post(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        logger.warning(f"Telegram API timeout on /{method}")
        return {}
    except Exception as e:
        logger.error(f"Telegram API error /{method}: {e}")
        return {}

def send_text(chat_id: int, text: str, reply_to: Optional[int] = None, parse_mode: str = "Markdown") -> dict:
    payload = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": parse_mode,
    }
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    return tg("sendMessage", json=payload)

def send_voice(chat_id: int, audio_path: Path, caption: Optional[str] = None) -> dict:
    with open(audio_path, "rb") as f:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption[:1024]
        return tg("sendVoice", data=data, files={"voice": f})

def send_typing(chat_id: int) -> None:
    tg("sendChatAction", json={"chat_id": chat_id, "action": "typing"})

def send_record_audio(chat_id: int) -> None:
    tg("sendChatAction", json={"chat_id": chat_id, "action": "record_voice"})

def download_file(file_id: str, dest_path: str) -> bool:
    """Download a Telegram file by file_id to dest_path."""
    try:
        file_info = tg("getFile", json={"file_id": file_id})
        file_path = file_info.get("result", {}).get("file_path", "")
        if not file_path:
            return False
        url  = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"File download failed (file_id={file_id}): {e}")
        return False

def notify_owner(message: str) -> None:
    """Send a notification message to the owner."""
    send_text(OWNER_ID, message, parse_mode="Markdown")

# ── Access Control ────────────────────────────────────────────────────────────
def is_authorized(chat_id: int, user_id: int, chat_type: str) -> bool:
    """Check if a user/chat is allowed to use the bot."""
    if user_id == OWNER_ID:
        return True
    if ALLOWED_USERS and user_id in ALLOWED_USERS:
        return True
    if chat_type in ("group", "supergroup") and ALLOWED_GROUPS and chat_id in ALLOWED_GROUPS:
        return True
    # If no explicit lists configured (beyond owner), default allow
    if not ALLOWED_USERS and not ALLOWED_GROUPS:
        return True
    return False

# ── Sleep / Wake Detection ────────────────────────────────────────────────────
def check_sleep_wake(text: str, chat_id: int) -> Optional[str]:
    """
    Check if message is a sleep/wake command.
    Returns response string if command matched, else None.
    """
    clean = text.lower().strip().rstrip("!.,?")

    # Wake up
    if sleeping_chats[chat_id]:
        if any(wake in clean for wake in WAKE_TRIGGERS):
            sleeping_chats[chat_id] = False
            return "👁 Aye, I'm back, sir! What chaos are we getting into now? 😏"
        # Still sleeping — silently ignore
        return ""

    # Sleep
    if any(trigger in clean for trigger in SLEEP_TRIGGERS):
        sleeping_chats[chat_id] = True
        return (
            "😴 Alright sir, standing by. Whisper my name whenever you need me back. "
            "*(Adimma Kann has entered stealth mode)*"
        )

    return None  # not a sleep/wake command

# ── Command Handlers ──────────────────────────────────────────────────────────
def handle_start(chat_id: int, user: dict, chat_type: str) -> None:
    """Handle /start command. Notify owner of new private chat users."""
    user_id    = user.get("id")
    first_name = user.get("first_name", "Unknown")
    username   = user.get("username", "N/A")

    welcome = (
        f"👋 Vanakkam, *{first_name}*! I'm *Adimma Kann* — "
        f"your sharp-tongued, Kerala-bred AI assistant.\n\n"
        f"I understand English, Malayalam, and Manglish.\n"
        f"Talk to me by text or voice — I'll reply the same way.\n\n"
        f"Type /help to see what I can do. 😏"
    )
    send_text(chat_id, welcome)

    if chat_type == "private" and user_id != OWNER_ID:
        notify_owner(
            f"🔔 *New user started the bot!*\n"
            f"  • Name     : {first_name}\n"
            f"  • Username : @{username}\n"
            f"  • User ID  : `{user_id}`"
        )

def handle_help(chat_id: int) -> None:
    """Handle /help or /instruction command."""
    help_text = (
        "🤖 Adimma Kann — Usage Guide\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📝 Text Messages\n"
        "Just type anything — English, Malayalam, or Manglish. I'll reply in the same language.\n\n"
        "🎤 Voice Messages\n"
        "Send a voice note. I'll transcribe and reply with voice + text.\n\n"
        "📸 Photos / Images\n"
        "Send an image (with or without caption). I'll describe it and respond.\n\n"
        "📄 Documents / PDFs\n"
        "Send a PDF or text file. I'll read it and you can ask questions.\n\n"
        "🌤 Weather\n"
        "Ask 'weather in Kochi' or 'what's the weather in Dubai?' — I'll fetch it live.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "💬 Commands\n"
        "/start — Start the bot\n"
        "/help or /instruction — This help message\n"
        "/clear — Clear conversation history\n"
        "/status — Check bot status\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "😴 Sleep / Wake\n"
        "Say bye / standby / sleep / good night → I go silent\n"
        "Say hi / hello / wake up / adimma → I come back\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🌐 Languages\n"
        "English 🇬🇧 | Malayalam 🇮🇳 | Manglish 😄\n"
    )
    send_text(chat_id, help_text)

def handle_clear(chat_id: int) -> None:
    """Handle /clear command — wipe conversation history."""
    conversation_history[chat_id].clear()
    send_text(
        chat_id,
        "🧹 Done, sir. Your conversation history has been wiped. Fresh start!\n"
        "(I still remember your personality though — unfortunately.) 😄"
    )

def handle_status(chat_id: int) -> None:
    """Handle /status command — show bot health info."""
    history_len  = len(conversation_history[chat_id])
    sleeping     = sleeping_chats[chat_id]
    keys_count   = len(GROQ_API_KEYS)
    cache_files  = len(list(TTS_CACHE_DIR.glob("*.mp3")))
    weather_ok   = bool(OPENWEATHER_API_KEY)

    status_text = (
        f"⚡ *Adimma Kann — Status*\n\n"
        f"  🔑 Groq API Keys  : `{keys_count}` loaded\n"
        f"  💬 History length : `{history_len}` messages\n"
        f"  😴 Sleep mode     : `{'Yes' if sleeping else 'No'}`\n"
        f"  🗣 TTS cache      : `{cache_files}` files\n"
        f"  🌤 Weather API    : `{'✅' if weather_ok else '❌ Not configured'}`\n"
        f"  🤖 LLM model      : `{LLM_MODEL}`\n"
        f"  👁 Vision model   : `{VISION_MODEL}`\n"
        f"  🎤 STT model      : `{STT_MODEL}`\n"
        f"  🕐 Server time    : `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`"
    )
    send_text(chat_id, status_text)

# ── Core Message Processors ───────────────────────────────────────────────────
def process_text_message(chat_id: int, text: str, message_id: int) -> None:
    """Process a plain text message."""
    send_typing(chat_id)

    language = detect_language(text)
    logger.info(f"[{chat_id}] Text [{language}]: {text[:80]}")

    reply_text = chat_with_groq(chat_id, text, language)
    send_text(chat_id, reply_text, reply_to=message_id)

    send_record_audio(chat_id)
    audio_path = synthesize_speech(reply_text, language)
    if audio_path:
        send_voice(chat_id, audio_path)

def process_voice_message(chat_id: int, voice: dict, message_id: int) -> None:
    """Process a voice message: STT → LLM → TTS."""
    send_typing(chat_id)

    file_id = voice.get("file_id", "")
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not download_file(file_id, tmp_path):
            send_text(chat_id, "❌ Couldn't download your voice message, sir. Try again?")
            return

        send_text(chat_id, "🎤 _Transcribing your voice..._", parse_mode="Markdown")
        transcript, stt_lang = transcribe_audio(tmp_path)

        if not transcript:
            send_text(chat_id, "🤔 Hmm, I couldn't make out what you said. Speak a bit clearer, sir?")
            return

        language = detect_language(transcript)
        if stt_lang == "ml":
            language = "malayalam"

        logger.info(f"[{chat_id}] Voice transcript [{language}]: {transcript[:80]}")
        send_text(
            chat_id,
            f"🎙 *You said:*\n_{transcript}_",
            parse_mode="Markdown"
        )

        reply_text = chat_with_groq(chat_id, transcript, language)
        send_text(chat_id, reply_text)

        send_record_audio(chat_id)
        audio_path = synthesize_speech(reply_text, language)
        if audio_path:
            send_voice(chat_id, audio_path)

    except Exception as e:
        logger.error(f"Voice processing error: {e}\n{traceback.format_exc()}")
        send_text(chat_id, "😬 Voice processing failed on my end, sir. Try texting it?")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def process_photo_message(chat_id: int, photos: list, caption: Optional[str], message_id: int) -> None:
    """Process a photo: describe via vision LLM → chat reply."""
    send_typing(chat_id)

    photo   = photos[-1]  # highest resolution
    file_id = photo.get("file_id", "")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not download_file(file_id, tmp_path):
            send_text(chat_id, "❌ Couldn't download the image, sir.")
            return

        send_text(chat_id, "👁 _Analysing the image..._", parse_mode="Markdown")
        description = describe_image(tmp_path)

        user_text = f"[Image received]\nImage description: {description}"
        if caption:
            user_text += f"\nUser's caption/question: {caption}"
            language   = detect_language(caption)
        else:
            user_text += "\nUser sent this image without any caption."
            language   = "english"

        reply_text = chat_with_groq(chat_id, user_text, language, extra_context=description)
        send_text(chat_id, f"🖼 *Image Analysis:*\n{reply_text}", reply_to=message_id)

        send_record_audio(chat_id)
        audio_path = synthesize_speech(reply_text, language)
        if audio_path:
            send_voice(chat_id, audio_path)

    except Exception as e:
        logger.error(f"Photo processing error: {e}")
        send_text(chat_id, "📸 Image analysis hit a snag, sir. Try again?")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def process_document_message(chat_id: int, document: dict, caption: Optional[str], message_id: int) -> None:
    """Process a document (PDF or text file)."""
    send_typing(chat_id)

    file_id   = document.get("file_id", "")
    file_name = document.get("file_name", "document")
    mime_type = document.get("mime_type", "")

    supported = (
        "pdf"  in mime_type.lower() or
        "text" in mime_type.lower() or
        file_name.lower().endswith((".pdf", ".txt", ".md", ".csv"))
    )
    if not supported:
        send_text(chat_id, f"📎 I can read PDFs and text files, sir. This appears to be a `{mime_type}` file — not supported yet.")
        return

    suffix = ".pdf" if ("pdf" in mime_type.lower() or file_name.lower().endswith(".pdf")) else ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not download_file(file_id, tmp_path):
            send_text(chat_id, "❌ Couldn't download the document, sir.")
            return

        send_text(chat_id, f"📄 _Reading {file_name}..._", parse_mode="Markdown")

        if suffix == ".pdf":
            content = extract_pdf_text(tmp_path)
        else:
            with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(4000)

        language  = detect_language(caption) if caption else "english"
        user_text = f"[Document: {file_name}]\n\nContent:\n{content}"
        if caption:
            user_text += f"\n\nUser's question/note: {caption}"
        else:
            user_text += "\n\nPlease summarise this document concisely."

        reply_text = chat_with_groq(chat_id, user_text, language, extra_context=content[:500])
        send_text(chat_id, f"📋 *Document Summary:*\n\n{reply_text}", reply_to=message_id)

        send_record_audio(chat_id)
        audio_path = synthesize_speech(reply_text, language)
        if audio_path:
            send_voice(chat_id, audio_path)

    except Exception as e:
        logger.error(f"Document processing error: {e}")
        send_text(chat_id, "📄 Had trouble reading that document, sir. Might be scanned/image-based.")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ── Main Update Handler ───────────────────────────────────────────────────────
def handle_update(update: dict) -> None:
    """Route an incoming Telegram update to the correct handler."""
    message = update.get("message") or update.get("edited_message")
    if not message:
        return

    chat_id    = message["chat"]["id"]
    chat_type  = message["chat"]["type"]
    message_id = message["message_id"]
    user       = message.get("from", {})
    user_id    = user.get("id", 0)

    # Access control
    if not is_authorized(chat_id, user_id, chat_type):
        logger.warning(f"Unauthorized access: user={user_id}, chat={chat_id}")
        send_text(chat_id, "🚫 Access denied. This bot is for authorised users only.")
        return

    text    = message.get("text", "").strip()
    caption = message.get("caption", "").strip() or None

    # ── Commands ──────────────────────────────────────────────────────────────
    if text.startswith("/start"):
        handle_start(chat_id, user, chat_type)
        return
    if text.startswith("/help") or text.startswith("/instruction"):
        handle_help(chat_id)
        return
    if text.startswith("/clear"):
        handle_clear(chat_id)
        return
    if text.startswith("/status"):
        handle_status(chat_id)
        return

    # ── Sleep/Wake check for text ─────────────────────────────────────────────
    if text:
        sleep_wake_response = check_sleep_wake(text, chat_id)
        if sleep_wake_response is not None:
            if sleep_wake_response:
                send_text(chat_id, sleep_wake_response)
            return

    # ── Skip if bot is sleeping ───────────────────────────────────────────────
    if sleeping_chats[chat_id]:
        return

    # ── Handle media types ────────────────────────────────────────────────────
    if "voice" in message or "audio" in message:
        voice = message.get("voice") or message.get("audio", {})
        process_voice_message(chat_id, voice, message_id)

    elif "photo" in message:
        process_photo_message(chat_id, message["photo"], caption, message_id)

    elif "document" in message:
        process_document_message(chat_id, message["document"], caption, message_id)

    elif text:
        process_text_message(chat_id, text, message_id)

    elif caption:
        process_text_message(chat_id, caption, message_id)

    else:
        send_text(chat_id, "🤷 I'm not sure what to do with that, sir. Send text, voice, image, or PDF.")

# ── Flask App & Webhook ───────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return jsonify({
        "status":    "online",
        "bot":       "Adimma Kann",
        "version":   "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "groq_keys": len(GROQ_API_KEYS)})


# ─────────────────────────────────────────────────────────────────────────────
# FIX: webhook() is now a plain sync function.
#
# The original code had no async issue here, but some deployment guides
# suggest making it async — that breaks standard Flask/Gunicorn without
# special async worker configuration (e.g. gevent or hypercorn).
#
# This version is safe with the default Gunicorn sync worker used on Render.
# ─────────────────────────────────────────────────────────────────────────────
@app.route(f"/webhook/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def webhook():
    """Receive Telegram webhook updates (sync — safe with Gunicorn sync workers)."""
    try:
        update = request.get_json(force=True, silent=True)
        if not update:
            return jsonify({"error": "No JSON body"}), 400

        logger.debug(f"Update received: {json.dumps(update)[:300]}")
        handle_update(update)
        return jsonify({"ok": True})

    except Exception as e:
        logger.error(f"Webhook error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ── Helper routes (useful for Render deployment) ──────────────────────────────
@app.route("/setwebhook", methods=["GET", "POST"])
def set_webhook_route():
    """
    Manually trigger webhook registration.
    Useful if auto-registration on startup failed (e.g. Render cold-start race).
    Visit: https://<your-render-url>/setwebhook
    """
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({"success": False, "error": "TELEGRAM_BOT_TOKEN not set"}), 400

    base_url = (
        request.args.get("url")          # ?url=https://... override
        or RENDER_EXTERNAL_URL
    )
    if not base_url:
        return jsonify({"success": False, "error": "RENDER_EXTERNAL_URL not set and no ?url= param"}), 400

    webhook_url = f"{base_url.rstrip('/')}/webhook/{TELEGRAM_BOT_TOKEN}"
    resp = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
        json={"url": webhook_url, "drop_pending_updates": True, "allowed_updates": ["message", "edited_message"]},
        timeout=10,
    )
    result = resp.json()
    logger.info(f"Manual setWebhook → {webhook_url}: {result}")
    return jsonify({"success": result.get("ok", False), "webhook_url": webhook_url, "telegram_response": result})


@app.route("/webhookinfo", methods=["GET"])
def webhook_info():
    """Check current Telegram webhook registration status."""
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 400
    resp = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo",
        timeout=10,
    )
    return jsonify(resp.json())


# ── Auto-register webhook on startup ─────────────────────────────────────────
def set_webhook() -> bool:
    """Register the webhook URL with Telegram on app startup."""
    if not RENDER_EXTERNAL_URL:
        logger.warning("RENDER_EXTERNAL_URL not set — webhook not registered automatically.")
        return False

    webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/webhook/{TELEGRAM_BOT_TOKEN}"
    resp = tg("setWebhook", json={
        "url":                  webhook_url,
        "drop_pending_updates": True,
        "allowed_updates":      ["message", "edited_message"],
    })
    ok = resp.get("ok", False)
    logger.info(f"Webhook {'registered ✅' if ok else 'FAILED ❌'}: {webhook_url}")
    return ok

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("═══ Adimma Kann Starting ═══")
    logger.info(f"Groq API keys  : {len(GROQ_API_KEYS)}")
    logger.info(f"Owner ID       : {OWNER_ID}")
    logger.info(f"LLM Model      : {LLM_MODEL}")
    logger.info(f"Vision Model   : {VISION_MODEL}")
    logger.info(f"STT Model      : {STT_MODEL}")
    logger.info(f"Allowed Users  : {ALLOWED_USERS}")
    logger.info(f"Allowed Groups : {ALLOWED_GROUPS}")

    set_webhook()

    # In production, Gunicorn runs this; locally we use Flask dev server
    app.run(host="0.0.0.0", port=PORT, debug=False)
