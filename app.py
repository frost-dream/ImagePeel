import os
import uuid
import cv2
import mimetypes
import requests
import base64
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from kokoro_onnx import Kokoro
import tempfile
import gc
import io
import smtplib
from email.mime.text import MIMEText
import re
from dotenv import load_dotenv
from pathlib import Path
from threading import Semaphore
from onnxruntime import InferenceSession, SessionOptions
from flask import Flask, request, render_template, render_template_string, Response, g, jsonify, send_from_directory
from PIL import Image
from shutil import rmtree
from random import randint
import sqlite3
from json import loads
from bcrypt import gensalt, hashpw, checkpw
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, unset_jwt_cookies
from datetime import timedelta, timezone, datetime
import secrets
import string
from time import time

app = Flask(__name__)

load_dotenv()

app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET')
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)

jwt = JWTManager(app)

brevo_api = os.getenv("BREVO_API_KEY")
brevo_login = os.getenv("BREVO_LOGIN")

salt = gensalt()

UPLOAD_FOLDER = 's'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png"}

MAX_CONCURRENT_JOBS = 10
processing_semaphore = Semaphore(MAX_CONCURRENT_JOBS)

MODEL_PATHS = {
    "removal": "models/model.onnx",
    "cartoon": "models/AnimeGANv3_Hayao_36.onnx",
    "upscale": "models/4xNomos8kDAT.onnx",
    "colorize": "models/colorizer_siggraph17.onnx",
}

famous_providers = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 
    'icloud.com', 'aol.com', 'protonmail.com', 'zoho.com', 
    'yandex.com', 'mail.com'
]

kokoro = Kokoro("models/kokoro-v1.0.int8.onnx", "models/voices-v1.0.bin")

_sessions = {}

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect('app.db')
        db.row_factory = sqlite3.Row
    return db

def find_user_by_identifier(identifier):
    db = get_db()
    cur = db.execute(
        "SELECT * FROM users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE",
        (identifier, identifier),
    )
    return cur.fetchone()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def get_session(key):
    if key in _sessions:
        return _sessions[key]
    if key not in MODEL_PATHS:
        raise RuntimeError("Unknown model key: " + str(key))
    path = MODEL_PATHS[key]
    sess_opt = SessionOptions()
    sess_opt.intra_op_num_threads = max(1, min(2, os.cpu_count() or 1))
    sess_opt.inter_op_num_threads = 1
    _sessions[key] = InferenceSession(path, sess_opt)
    return _sessions[key]

def safe_tempfile(suffix=""):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.close()
    return f.name

def free_mem(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()

def to_multiple_of_8(x):
    if x % 8 == 0:
        return x
    return x + (8 - (x % 8))

def pil_to_numpy_rgb(pil_img):
    return np.array(pil_img.convert("RGB"))

def numpy_rgb_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))

def tile_process_bgr(img_bgr, run_single_tile_fn, tile_size=512, pad_to_multiple=8):
    h, w = img_bgr.shape[:2]
    out = np.zeros_like(img_bgr, dtype=np.uint8)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1 = y
            x1 = x
            y2 = min(h, y + tile_size)
            x2 = min(w, x + tile_size)
            tile = img_bgr[y1:y2, x1:x2]
            try:
                tile_out = run_single_tile_fn(tile)
            except Exception:
                tile_out = cv2.resize(tile, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LINEAR)
            if tile_out.shape[:2] != tile.shape[:2]:
                tile_out = cv2.resize(tile_out, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LINEAR)
            out[y1:y2, x1:x2] = np.clip(tile_out, 0, 255).astype(np.uint8)
    return out

def image_removal(path):
    session = get_session("removal")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pil_img = Image.open(path).convert("RGB")
    orig_w, orig_h = pil_img.size

    target = 1024
    resized = pil_img.resize((target, target))
    arr = np.array(resized).astype(np.float32) / 255.0
    inp = np.transpose(arr, (2, 0, 1))[None, :]

    try:
        mask = session.run([output_name], {input_name: inp})[0][0, 0]
    finally:
        free_mem(arr, inp, resized)

    mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h))
    rgb = pil_img
    rgba = Image.new("RGBA", rgb.size)
    rgba.paste(rgb)
    rgba.putalpha(mask_img)
    out_path = os.path.join(path.rsplit('/' if '/' in path else '\\', 1)[0], str(uuid.uuid4()) + '.png')
    rgba.save(out_path, optimize=True)
    free_mem(mask_img, rgba, rgb)
    return out_path

def image_cartoon(path):
    session = get_session("cartoon")
    img = cv2.imread(path).astype(np.float32)
    to_8s = lambda x: (256 if x < 256 else x - x % 8)
    new_path = os.path.join(path.rsplit('/' if '/' in path else '\\', 1)[0], str(uuid.uuid4()) + '.' + path.rsplit('.', 1)[1])
    cv2.imwrite(new_path, cv2.cvtColor(
        cv2.resize(
            np.clip(
                (np.squeeze(
                    session.run(
                        None, {session.get_inputs()[0].name:
                               np.expand_dims(
                                   cv2.cvtColor(
                                       cv2.resize(img, (to_8s(img.shape[1]), to_8s(img.shape[0]))),
                                       cv2.COLOR_BGR2RGB
                                   ).astype(np.float32) / 127.5 - 1.0,
                                   axis=0
                               )
                        }
                    )[0]
                ) + 1.) / 2 * 255,
                0, 255
            ).astype(np.uint8),
            (img.shape[1], img.shape[0])
        ),
        cv2.COLOR_RGB2BGR
    ))
    free_mem(img)
    return new_path


def image_upscale(path, x):
    session = get_session('upscale')

    img = Image.open(path).convert("RGB")
    width, height = img.size

    scale_map = {"2": 1.2, "3": 1.5, "4": 2.0}
    scale = scale_map.get(str(x), 1.2)
    new_w = int(min(width * scale, 2048))
    new_h = int(min(height * scale, 2048))

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    inp = np.array(img_resized, dtype=np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, :]

    output = session.run(
        None, {session.get_inputs()[0].name: inp}
    )[0]

    out = np.clip(output[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    new_path = os.path.join(path.rsplit('/' if '/' in path else '\\', 1)[0], str(uuid.uuid4()) + '.' + path.rsplit('.', 1)[1])
    Image.fromarray(out).save(new_path, optimize=True)

    del img, img_resized, inp, output, out
    gc.collect()
    return new_path

def colorize(path):
    session = get_session("colorize")
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError("Failed to read image")
    if img_bgr.ndim == 2:
        img_bgr = np.tile(img_bgr[:, :, None], (1, 1, 3))
    img_lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    L_orig = img_lab[:, :, 0]
    small = cv2.resize(img_bgr, (320, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    small_lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
    inp = np.expand_dims(np.expand_dims(small_lab[:, :, 0], 0), 0).astype(np.float32)

    out_ab = session.run(None, {"input": inp})[0]
    out_ab = out_ab[0].transpose(1, 2, 0)
    out_ab_up = cv2.resize(out_ab, (L_orig.shape[1], L_orig.shape[0]), interpolation=cv2.INTER_LINEAR)
    lab_merge = cv2.merge([L_orig, out_ab_up[:, :, 0], out_ab_up[:, :, 1]])
    bgr = (cv2.cvtColor(lab_merge, cv2.COLOR_Lab2BGR) * 255.0).clip(0, 255).astype(np.uint8)
    new_path = os.path.join(path.rsplit('/' if '/' in path else '\\', 1)[0], str(uuid.uuid4()) + '.' + path.rsplit('.', 1)[1])
    cv2.imwrite(new_path, bgr)
    free_mem(img_bgr, small, inp, out_ab, out_ab_up, lab_merge)
    return new_path

@app.route('/')
def index():
    # request.headers.get('CF-Connecting-IP')
    return render_template('index.html')

@app.route('/typing')
def typing():
    return render_template('typing.html')

@app.route("/api/<action>", methods=["POST"])
@jwt_required(optional=True)
def upload_file(action):
    out = path_of_user = None
    acquired = processing_semaphore.acquire(blocking=False)
    if not acquired:
        return {"error": "Server busy, try again later"}, 429
    try:
        if action == 'txt2img':
            body = loads(request.data or b'{}')
            negative = body.get('negative-prompt', '')
            prompt = body.get('prompt') or 'A stunning hyperrealistic photograph of an astronaut on Mars'
            width = str(int(body.get('width', 512)) - (int(body.get('width', 512)) % 8))
            height = str(int(body.get('height', 512)) - (int(body.get('height', 512)) % 8))
            if int(width) * int(height) > 400000:
                return {"error": "Requested image too large"}, 400
            steps = body.get('steps', '1')
            cfg = body.get('cfg', '7.5')
            seed = body.get('seed', randint(-1, 100000))
            filepath = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '.png')
            resp = requests.post("http://127.0.0.1:8000/api/generate", json={
                "prompt": prompt,
                "negative_prompt": negative,
                "image_width": width,
                "image_height": height,
                "inference_steps": steps,
                "guidance_scale": cfg,
                "seed": seed,
            }, timeout=300)
            try:
                data = resp.json() 
                img_b64 = data['images'][0]
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(img_b64))
            except Exception as e:
                return {"error": "txt2img proxy failed", "details": str(e)}, 500

        elif action == 'tts':
            uuidpath = str(uuid.uuid4())
            filepath = os.path.join(UPLOAD_FOLDER, uuidpath + '.wav')
            form = request.form
            def stream_audio():
                with open('static/imagepeel.com.mp3', 'rb') as fh:
                    while True:
                        chunk = fh.read(16 * 1024)
                        if not chunk:
                            break
                        yield chunk
            if form['prompt']:
                def stream_and_remove_file_audio(fp):
                    with open(fp, 'rb') as fh:
                        while True:
                            chunk = fh.read(16 * 1024)
                            if not chunk:
                                break
                            yield chunk
                    os.remove(fp)
                voice = form['voice']
                if form['voice-blend']:
                    first_voice = kokoro.get_voice_style(voice)
                    second_voice = kokoro.get_voice_style(form['voice-blend'])
                    voice = np.add(first_voice * (50 / 100), second_voice * (50 / 100))
                samples, sample_rate = kokoro.create(form['prompt'], voice=voice, speed=float(form['speed']))
                sf.write(filepath, samples, sample_rate)
                audio = AudioSegment.from_wav(filepath)
                os.remove(filepath)
                filepath = os.path.join(UPLOAD_FOLDER, uuidpath + '.mp3')
                audio.export(filepath, format="mp3")
                return Response(stream_and_remove_file_audio(filepath), headers={'Content-Disposition': f'attachment"'}, mimetype='application/octet-stream')
            else:
                return Response(stream_audio(), headers={'Content-Disposition': f'attachment"'}, mimetype='application/octet-stream')

        else:
            if "file" not in request.files:
                url = loads(request.data)
                if not url:
                    return {"error": "No input"}, 400
                user_id = get_jwt_identity()
                path_of_user = filepath = os.path.join(UPLOAD_FOLDER, user_id, url[2:])
            else:
                filepath = None
                try:
                    user_id = get_jwt_identity()
                    path_of_user = filepath = os.path.join(UPLOAD_FOLDER, user_id)
                except:
                    pass
                file = request.files["file"]
                mime_type = file.mimetype
                if mime_type not in ALLOWED_TYPES:
                    return {"error": "Only image allowed"}, 400
                file_id = str(uuid.uuid4())
                ext = mimetypes.guess_extension(mime_type) or ""
                filename = f"{file_id}{ext}"
                if filepath:
                    filepath = os.path.join(filepath, filename)
                else:
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                target_bytes = 100 * 1024
                img = Image.open(filepath)
                width, height = img.size
                low, high = 0.1, 1.0
                best_img = img
                while high - low > 0.01:
                    mid = (low + high) / 2
                    new_w = int(width * mid)
                    new_h = int(height * mid)
                    resized = img.resize((new_w, new_h), Image.LANCZOS)
                    buffer = io.BytesIO()
                    resized.save(buffer, format=img.format)
                    size = buffer.tell()
                    if size > target_bytes:
                        high = mid
                    else:
                        low = mid
                        best_img = resized
                best_img.save(filepath, format=img.format)

            if action == 'removal':
                filepath, out = image_removal(filepath), filepath

            elif action == 'cartoon':
                filepath, out = image_cartoon(filepath), filepath

            elif action == 'upscale':
                x = request.form.get('x', request.files.get('x', '2'))
                filepath, out = image_upscale(filepath, x), filepath

            elif action == 'colorize':
                filepath, out = colorize(filepath), filepath

            else:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return {"error": "Invalid request"}, 400

        def stream_and_remove_file(fp, out):
            try:
                with open(fp, 'rb') as fh:
                    while True:
                        chunk = fh.read(16 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    if not path_of_user:
                        os.remove(fp)
                        os.remove(out)
                except Exception:
                    pass
        return Response(stream_and_remove_file(filepath, out), headers={'Content-Disposition': f'attachment"'}, mimetype='application/octet-stream')

    finally:
        processing_semaphore.release()
        free_mem()
        

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'GET':
        with open('feedback.txt', 'r') as file:
            file = file.read()
        g = file.count('g')
        n = file.count('n')
        b = file.count('b')
        s = g + n + b
        if s == 0:
            s = 1
        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Feedback Overview</title>
        <style> body {{ font-family: Arial, sans-serif; background: #f3f4f6; display:flex;justify-content:center;align-items:center;height:100vh }}
        .card{{background:white;padding:30px;border-radius:16px;box-shadow:0 8px 20px rgba(0,0,0,0.1);width:400px;text-align:center}}
        .chart{{position:relative;height:200px;display:flex;justify-content:space-around;align-items:flex-end}}
        .bar{{width:60px;border-radius:8px 8px 0 0;display:flex;align-items:flex-end;justify-content:center;color:white;font-weight:bold}}
        .bar.good{{background:#22c55e;height:{int((g/s)*100)}px}}
        .bar.normal{{background:#facc15;height:{int((n/s)*100)}px}}
        .bar.bad{{background:#ef4444;height:{int((b/s)*100)}px}}</style></head><body>
        <div class="card"><div class="chart"><div class="bar good">{g}</div><div class="bar normal">{n}</div><div class="bar bad">{b}</div></div>
        <div style="display:flex;justify-content:space-around;margin-top:10px;font-weight:bold;color:#374151"><span>Good 👍</span><span>Normal 😐</span><span>Bad 👎</span></div></div></body></html>"""
        return render_template_string(html)
    else:
        with open('feedback.txt', 'a') as file:
            if request.data == b'Good':
                file.write('g')
            elif request.data == b'Normal':
                file.write('n')
            elif request.data == b'Bad':
                file.write('b')
        return 'OK', 200

@app.route('/api/subscribe', methods=['POST'])
def sub():
    email = request.data.decode()
    if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email) is None:
        return "format"
    if email.rsplit('@', 1)[1] not in famous_providers:
        return "accept"
    with open('email.txt', 'r') as file:
        file = file.read()
    if email in file.splitlines():
        return "already"
    data = f'{email} {secrets.token_urlsafe(30)}'
    with open('email.txt', 'a') as file:
        file.write('\n' + data)
    send_email(*data.split(' '))
    return 'null'

def send_email_verify(email, code):
    msg = MIMEText(f'''
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 24px; background: #fafafa; border-radius: 12px; border: 1px solid #ddd;">
  <img src="https://imagepeel.com/static/icon.png" width="60" style="display:block;margin:0 auto 16px;">
  <h2 style="color: #4A4A4A; text-align: center;">👋 Welcome to ImagePeel!</h2>
  <p style="font-size: 16px; color: #333;">
                   Welcome to Image Peel, <strong>{code}</strong> is your verification code.
                   \n\n
                   This code expires in 10 minutes.
  </p>

  <p style="font-size: 14px; color: #999; margin-top: 16px; text-align: center;">
    © 2025 ImagePeel. All rights reserved.
  </p>
</div>

''', "html")
    msg["Subject"] = 'Verify your account'
    msg["From"] = "System <system@imagepeel.com>"
    msg["To"] = email
    with smtplib.SMTP("smtp-relay.brevo.com", 587) as server:
        server.login(brevo_login, brevo_api)
        server.send_message(msg)

def send_email(recipient, unsub, news=True):
    if news:
        news = '''
  <p style="font-size: 16px; color: #333;">
    You’ll receive updates about:
    <ul style="margin-top: 8px; margin-bottom: 8px;">
      <li>🧠 New AI tools and experiments on ImagePeel</li>
      <li>📰 AI news and cool discoveries</li>
      <li>📊 Occasional surveys (so you can shape our next tool)</li>
    </ul>
  </p>

  <p style="font-size: 16px; color: #333;">
    We usually send one email per week — sometimes less, never spam.
  </p>
'''
    else:
        news = ''
    msg = MIMEText(f'''
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 24px; background: #fafafa; border-radius: 12px; border: 1px solid #ddd;">
  <img src="https://imagepeel.com/static/icon.png" width="60" style="display:block;margin:0 auto 16px;">
  <h2 style="color: #4A4A4A; text-align: center;">👋 Welcome to ImagePeel!</h2>
  <p style="font-size: 16px; color: #333;">
    Thanks for signing up! You’re now part of the ImagePeel community — where we turn AI ideas into real, fun tools.
  </p>
{news}
  <p style="font-size: 14px; color: #666; margin-top: 32px;">
    Didn’t sign up? No worries — you can 
    <a href="https://imagepeel.com/unsubscribe/{unsub}" style="color: #0066ff; text-decoration: none;">unsubscribe here</a>.
  </p>

  <p style="font-size: 14px; color: #999; margin-top: 16px; text-align: center;">
    © 2025 ImagePeel. All rights reserved.
  </p>
</div>

''', "html")
    msg["Subject"] = 'Welcome to ImagePeel — your AI toolkit!'
    msg["From"] = "Image Peel <welcome@imagepeel.com>"
    msg["To"] = recipient
    with smtplib.SMTP("smtp-relay.brevo.com", 587) as server:
        server.login(brevo_login, brevo_api)
        server.send_message(msg)

def send_email_reset(email, user, code):
    msg = MIMEText(f'''
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 24px; background: #fafafa; border-radius: 12px; border: 1px solid #ddd;">
  <img src="https://imagepeel.com/static/icon.png" width="60" style="display:block;margin:0 auto 16px;">
  <h2 style="color: #4A4A4A; text-align: center;">Your reset password request</h2>
    <p style="font-size: 16px; color: #333;">
        Hello {user},\n\nYour password reset code is: {code}\n\nUse this code on the reset screen to update your password.\nIf you didn't request this, ignore this email.
        \n\n
        This code expires in 10 minutes.
    </p>
    <p style="font-size: 14px; color: #999; margin-top: 16px; text-align: center;">
        © 2025 ImagePeel. All rights reserved.
    </p></div>''')
    msg["Subject"] = 'Your reset password code'
    msg["From"] = "System <system@imagepeel.com>"
    msg["To"] = email
    with smtplib.SMTP("smtp-relay.brevo.com", 587) as server:
        server.login(brevo_login, brevo_api)
        server.send_message(msg)

@app.route('/unsubscribe/<sub>/<email>')
def unsubscribe(sub, email):
    db = get_db()
    db.execute("SELECT 1 FROM users WHERE email = ? AND token = ? LIMIT 1", (email, sub))
    if db.fetchone():
        user = find_user_by_identifier(email)
        rmtree(user['id'])
        db.execute("DELETE FROM users WHERE email = ? AND token = ?", (email, sub))
        db.commit()
        return 'Success!', 200
    else:
        return 'Not found', 404


@app.route('/join', methods=['POST', 'GET'])
def join():
    if request.method == 'GET':
        return render_template('join.html')
    else:
        form = request.form
        match form.get('method'):
            case 'login':
                identifier = (form.get("identifier") or "").strip()
                password = form.get("password", "")

                if not identifier or not password:
                    return ("Missing fields", 400)

                user = find_user_by_identifier(identifier)
                if not user:
                    return ("Invalid credentials", 400)

                if not checkpw(password.encode(), user["password_hash"]):
                    return ("Invalid credentials", 400)
                
                if user['is_verified'] == 'n':
                    return ("Account is not verified", 400)

                access_token = create_access_token(identity=str(user['id']))
                return jsonify({"access_token": access_token}), 200
            case 'signup':
                username = (form.get("username") or "").strip()
                email = (form.get("email") or "").strip().lower()
                password = form.get("password", "")

                if not username or not email or not password:
                    return ("Missing fields", 400)
                db = get_db()
                if db.execute("SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone():
                    return ("Username already taken", 400)
                if db.execute("SELECT 1 FROM users WHERE email = ? COLLATE NOCASE", (email,)).fetchone():
                    return ("Email already registered", 400)

                password_hash = hashpw(password.encode(), salt)
                token = secrets.token_urlsafe(30)
                created_at = datetime.now(timezone.utc).isoformat()
                code = "".join(secrets.choice(string.digits) for _ in range(6))
                db.execute(
                    "INSERT INTO users (username, email, password_hash, is_verified, token, code, expire, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (username, email, password_hash, 'n', token, code, int(time()) + 60 * 10, created_at),
                )
                db.commit()
                send_email_verify(email, code)
                return ("created", 200)
            case 'verify':
                code = (form.get("verificationCode") or "").strip()
                email = (form.get("email") or "").strip().lower()
                if not code or not email:
                    return ("Missing fields", 400)
                db = get_db()
                cur = db.execute("SELECT * FROM users WHERE code = ? COLLATE NOCASE", (code,))
                user = cur.fetchall()
                ret = False
                for i in user:
                    if int(i['expire']) > time():
                        db.execute("UPDATE users SET is_verified = ?, code = NULL, expire = NULL WHERE id = ?", ('y', i["id"]))
                        db.commit()
                        ret = True
                    else:
                        db.execute("UPDATE users SET code = NULL WHERE id = ?", (i["id"]))
                        db.commit()
                if ret:
                    os.mkdir(f'{UPLOAD_FOLDER}/{db.execute("SELECT id FROM users ORDER BY id DESC LIMIT 1").fetchone()[0]}')
                    return ("Verified", 200)
                return ("Code expired", 400)
            case 'forgot':
                identifier = (form.get("identifier") or "").strip()
                if not identifier:
                    return ("Missing identifier", 400)
                user = find_user_by_identifier(identifier)
                if not user:
                    return ("sent", 200)
                if (int(user['expire']) - int(time())) > 9 * 60:
                    return 'Too many requests', 429
                code = "".join(secrets.choice(string.digits) for _ in range(6))
                db = get_db()
                db.execute("UPDATE users SET code = ?, expire = ? WHERE id = ?",(code, int(time()) + 60 * 10, user["id"]))
                db.commit()

                send_email_reset(user["email"], user['username'], code)

                return ("sent", 200)
            case 'resend':
                identifier = (form.get("email") or "").strip()
                if not identifier:
                    return ("Missing identifier", 400)
                user = find_user_by_identifier(identifier)
                if not user:
                    return ("sent", 200)
                if (int(user['expire']) - int(time())) > 9 * 60:
                    return 'Too many requests', 429
                code = "".join(secrets.choice(string.digits) for _ in range(6))
                db = get_db()
                db.execute("UPDATE users SET code = ?, expire = ? WHERE id = ?",(code, int(time()) + 60 * 10, user["id"]))
                db.commit()

                send_email_reset(user["email"], user['username'], code)

                return ("sent", 200)
            case 'reset':
                code = (form.get("resetCode") or "").strip()
                new_password = form.get("newPassword", "")

                if not code or not new_password:
                    return ("Missing fields", 400)

                db = get_db()
                cur = db.execute("SELECT * FROM users WHERE code = ? COLLATE NOCASE", (code,))
                user = cur.fetchall()
                ret = False
                for i in user:
                    if int(i['expire']) > time():
                        new_hash = hashpw(new_password.encode(), salt)
                        db.execute("UPDATE users SET password_hash = ?, code = NULL, expire = NULL WHERE id = ?", (new_hash, i["id"]))
                        db.commit()
                        ret = True
                    else:
                        db.execute("UPDATE users SET code = NULL WHERE id = ?", (i["id"]))
                        db.commit()
                if ret:
                    return ("updated", 200)
                return ("Code expired", 400)
            case _:
                return ("Unknown method", 400)

@app.route("/api/logout", methods=["POST"])
def logout():
    response = jsonify({"message": "Logout successful"})
    unset_jwt_cookies(response)
    return response

@app.route("/api/history", methods=["GET"])
@jwt_required()
def get_user_images():
    user_id = get_jwt_identity()
    images = []
    directory = Path(f"s/{user_id}")
    files = [f for f in directory.iterdir()]
    files_sorted = sorted(files, key=lambda f: f.stat().st_mtime)
    for file in files_sorted:
        images.append({"output_url": f"s/{file.name}","timestamp": datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})
    return jsonify(images), 200

@app.route('/s/<path>')
@jwt_required()
def get_s(path):
    user_id = get_jwt_identity()
    return send_from_directory(f's/{user_id}', path)