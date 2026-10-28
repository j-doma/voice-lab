"""Local SpeechBrain enrollment/identification dashboard. Only record with consent."""
import json
import io
import wave
import threading
import webbrowser
from pathlib import Path
from fractions import Fraction
import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from flask import Flask, jsonify, request, render_template_string, send_file
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy


TARGET_RATE = 16000
MIN_SECONDS = 3
THRESHOLD = 0.50# Single-speaker threshold.
GROUP_THRESHOLD = 0.40  # Group matches; not a probability.
GROUP_UNKNOWN_THRESHOLD = 0.20
SECOND_TRACK_ENERGY_RATIO = 0.22  # Heuristic: reject very weak SepFormer ghost tracks.
ROOT = Path(__file__).resolve().parent
PROFILE_FILE = ROOT / 'voice_profiles.json'
SEPARATED_FILE = ROOT / 'voice_profiles_separated.json'
app = Flask(__name__)
lock = threading.RLock()
state = {'mode': None, 'chunks': [], 'rms': 0.0, 'peak': 0.0,
         'message': 'Loading model...', 'result': None, 'processing': False, 'enroll_name': None}
model = None
separator = None
last_separated = None
stream = None
sample_rate = TARGET_RATE


def load_profiles():
    if not PROFILE_FILE.exists():
        return {}
    raw = json.loads(PROFILE_FILE.read_text(encoding='utf-8'))
    return {name: [np.asarray(v, dtype=np.float32) for v in samples]
            for name, samples in raw.items()}


profiles = load_profiles()


def load_separated_profiles():
    if not SEPARATED_FILE.exists():
        return {}
    raw = json.loads(SEPARATED_FILE.read_text(encoding='utf-8'))
    return {name: [np.asarray(v, dtype=np.float32) for v in samples]
            for name, samples in raw.items()}


separated_profiles = load_separated_profiles()


def save_separated_profiles():
    data = {name: [v.tolist() for v in samples] for name, samples in separated_profiles.items()}
    SEPARATED_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')


def save_profiles():
    data = {name: [v.tolist() for v in samples] for name, samples in profiles.items()}
    PROFILE_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')


def callback(indata, frames, time_info, status):
    mono = np.asarray(indata, dtype=np.float32).mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))
    with lock:
        state['rms'] = rms
        state['peak'] = peak
        if state['mode']:
            state['chunks'].append(mono.copy())


def embedding(audio):
    if sample_rate != TARGET_RATE:
        ratio = Fraction(TARGET_RATE, sample_rate)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
    waveform = torch.from_numpy(audio.copy()).float().unsqueeze(0)
    with torch.no_grad():
        vec = model.encode_batch(waveform).squeeze().cpu().numpy().reshape(-1)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError('Invalid voice embedding')
    return (vec / norm).astype(np.float32)


def status_payload():
    with lock:
        return dict(mode=state['mode'], rms=state['rms'], peak=state['peak'],
                    message=state['message'], result=state['result'],
                    processing=state['processing'],
                    people=[{'name': k, 'samples': len(v), 'separated_samples': len(separated_profiles.get(k, []))} for k, v in sorted(profiles.items())])


def separated_embedding(voice8):
    voice16 = resample_poly(voice8, 2, 1).astype(np.float32)
    waveform = torch.from_numpy(voice16.copy()).unsqueeze(0)
    with torch.no_grad():
        vec = model.encode_batch(waveform).squeeze().cpu().numpy().reshape(-1)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError('Invalid separated voice embedding')
    return (vec / norm).astype(np.float32)


def separate_and_identify(audio):
    global separator, last_separated
    # Keep clean and SepFormer reference embeddings separate.
    if separator is None:
        with lock:
            state['message'] = 'Loading SepFormer (first download may be large)...'
        separator = SepformerSeparation.from_hparams(
            source='speechbrain/sepformer-wsj02mix',
            savedir=str(ROOT / 'pretrained_models' / 'sepformer-wsj02mix'),
            run_opts={'device': 'cpu'},
            local_strategy=LocalStrategy.COPY,
        )
    ratio = Fraction(8000, sample_rate)
    mixture = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
    with torch.no_grad():
        separated = separator.separate_batch(torch.from_numpy(mixture.copy()).unsqueeze(0))
    separated = separated.squeeze(0).detach().cpu().numpy()
    if separated.ndim != 2 or separated.shape[1] != 2:
        raise RuntimeError('SepFormer returned an unexpected audio shape.')
    with lock:
        last_separated = [separated[:, i].astype(np.float32).copy() for i in range(2)]
    tracks = last_separated
    energies = [float(np.sqrt(np.mean(track ** 2))) for track in tracks]
    strongest = max(energies)
    results = []
    for i, voice8 in enumerate(tracks):
        energy = energies[i]
        if energy < 0.001 or energy < strongest * SECOND_TRACK_ENERGY_RATIO:
            continue
        vec = separated_embedding(voice8)
        scores = []
        for person, clean_samples in profiles.items():
            sep_samples = separated_profiles.get(person, [])
            reference = sep_samples if sep_samples else clean_samples
            reference_type = 'separated' if sep_samples else 'normal (fallback)'
            if reference:
                score = float(np.mean([np.dot(vec, v) for v in reference]))
                scores.append((person, score, reference_type))
        scores.sort(key=lambda item: item[1], reverse=True)
        if not scores:
            results.append({'channel': i + 1, 'name': 'Unknown speaker', 'score': None})
            continue
        best, score, ref_type = scores[0]
        label = best if score >= GROUP_THRESHOLD else 'Unknown speaker'
        results.append({'channel': i + 1, 'name': label,
                        'closest_name': best, 'score': round(score, 3),
                        'reference': ref_type, 'energy': round(energy, 5)})

    # SepFormer can output the SAME voice twice. Don't count duplicated identity as two people.
    if len(results) == 2:
        first, second = results
        same_identity = first.get('closest_name') == second.get('closest_name')
        waveform_corr = float(np.dot(tracks[0], tracks[1]) /
                              max(np.linalg.norm(tracks[0]) * np.linalg.norm(tracks[1]), 1e-8))
        if (same_identity and max(first.get('score') or 0, second.get('score') or 0) >= GROUP_THRESHOLD) or abs(waveform_corr) > 0.85:
            results = [max(results, key=lambda r: (r.get('score') or -1, r.get('energy') or 0))]

    # If only one track identifies someone, prefer that speaker over a weak ghost output.
    if len(results) == 2:
        identified = [r for r in results if (r.get('score') or -1) >= GROUP_THRESHOLD]
        if len(identified) == 1:
            results = identified

    if not results:
        return {'kind': 'group', 'speakers': [], 'count': 0,
                'note': 'No clear voices detected.'}
    for result in results:
        score = result.get('score')
        if score is not None and score < GROUP_UNKNOWN_THRESHOLD:
            result['closest_name'] = None
    return {'kind': 'group', 'speakers': results, 'count': len(results),
            'note': 'Speaker count is estimated from separated tracks, not guaranteed.'}


@app.get('/group_audio/<int:channel>')
def group_audio(channel):
    if channel not in (1, 2):
        return jsonify(error='Invalid voice channel.'), 400
    with lock:
        if last_separated is None:
            return jsonify(error='No separated audio available.'), 404
        audio = last_separated[channel - 1].copy()
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2')
        wav.writeframes(pcm.tobytes())
    output.seek(0)
    return send_file(output, mimetype='audio/wav', download_name=f'voice_{channel}.wav')


@app.post('/save_separated')
def save_separated():
    data = request.get_json(silent=True) or {}
    name = str(data.get('name') or '').strip()
    channel = data.get('channel')
    if channel not in (1, 2):
        return jsonify(error='Choose Voice 1 or Voice 2.'), 400
    with lock:
        if state['mode'] or state['processing']:
            return jsonify(error='Wait for recording to finish.'), 409
        if name not in profiles:
            return jsonify(error='Choose an enrolled person.'), 400
        if last_separated is None:
            return jsonify(error='Record and separate voices first.'), 400
        voice = last_separated[channel - 1].copy()
        state['processing'] = True
    try:
        if float(np.sqrt(np.mean(voice ** 2))) < 0.001:
            raise ValueError('Selected separated track is too quiet.')
        vec = separated_embedding(voice)
        with lock:
            separated_profiles.setdefault(name, []).append(vec)
            save_separated_profiles()
            state['message'] = f'Saved separated voice sample for {name}.'
        return jsonify(status_payload())
    except Exception as exc:
        with lock:
            state['message'] = str(exc)
        return jsonify(error=str(exc)), 400
    finally:
        with lock:
            state['processing'] = False


@app.get('/status')


def get_status():
    return jsonify(status_payload())


@app.post('/start')


def start():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode')
    name = str(data.get('name') or '').strip()
    if mode not in ('enroll', 'identify', 'group'):
        return jsonify(error='Invalid mode'), 400
    if mode == 'enroll' and not name:
        return jsonify(error='Enter a name before starting.'), 400
    with lock:
        if state['mode'] or state['processing']:
            return jsonify(error='Stop the current recording/analysis first.'), 409
        if mode in ('identify', 'group') and not profiles:
            return jsonify(error='Enroll a person first.'), 400
        state['mode'] = mode
        if mode == 'group':
            global last_separated
            last_separated = None
        state['enroll_name'] = name if mode == 'enroll' else None
        state['chunks'] = []
        state['result'] = None
        state['message'] = ('Recording voice...' if mode == 'enroll' else 'Recording two-person mixture...' if mode == 'group' else 'Listening for identification...')
    return jsonify(status_payload())


@app.post('/stop')


def stop():
    data = request.get_json(silent=True) or {}
    with lock:
        mode = state['mode']
        name = state['enroll_name']
        if not mode:
            return jsonify(error='No recording active.'), 409
        if mode == 'enroll' and not name:
            return jsonify(error='Enter a name before saving.'), 400
        state['mode'] = None
        state['enroll_name'] = None
        audio = np.concatenate(state['chunks']) if state['chunks'] else np.array([], dtype=np.float32)
        state['chunks'] = []
        state['processing'] = True
        state['message'] = 'Analyzing audio...'
    try:
        seconds = len(audio) / sample_rate
        if seconds < MIN_SECONDS:
            raise ValueError(f'Record at least {MIN_SECONDS} seconds (got {seconds:.1f}s).')
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.002:
            raise ValueError(f'Audio too quiet (RMS {rms:.5f}). Check the microphone.')
        vec = embedding(audio) if mode != 'group' else None
        if mode == 'group':
            group_result = separate_and_identify(audio)
        with lock:
            if mode == 'enroll':
                profiles.setdefault(name, []).append(vec)
                save_profiles()
                state['result'] = {'kind': 'enroll', 'name': name}
                state['message'] = f'Saved {name} ({len(profiles[name])} sample(s)).'
            elif mode == 'group':
                state['result'] = group_result
                state['message'] = f"Estimated {group_result['count']} speaker(s). Review matches below."
            else:
                scores = sorted(((person, float(np.mean([np.dot(vec, v) for v in samples])))
                                 for person, samples in profiles.items() if samples),
                                key=lambda item: item[1], reverse=True)
                best_name, best_score = scores[0]
                guess = best_name if best_score >= THRESHOLD else 'Unknown speaker'
                state['result'] = {'kind': 'identify', 'name': guess,
                                   'scores': [{'name': p, 'score': round(s, 3)} for p, s in scores[:5]]}
                state['message'] = f'Guess: {guess} (similarity {best_score:.3f}; experimental)'
    except Exception as exc:
        with lock:
            state['message'] = str(exc)
            state['result'] = None
        return jsonify(error=str(exc), **status_payload()), 400
    finally:
        with lock:
            state['processing'] = False
    return jsonify(status_payload())


@app.post('/delete_profile')


def delete_profile():
    data = request.get_json(silent=True) or {}
    name = str(data.get('name') or '').strip()
    with lock:
        if state['mode'] or state['processing']:
            return jsonify(error='Stop recording/analysis before deleting.'), 409
        if name not in profiles:
            return jsonify(error='Profile not found.'), 404
        del profiles[name]
        separated_profiles.pop(name, None)
        save_separated_profiles()
        save_profiles()
        state['result'] = None
        state['message'] = f'Deleted {name}.'
    return jsonify(status_payload())


@app.post('/quit')


def quit_app():
    # Stop the local server process.
    import os
    threading.Timer(0.35, lambda: os._exit(0)).start()
    return jsonify(ok=True)


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Lab · SpeechBrain</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#243348;background:#f4f7fb}
*{box-sizing:border-box}body{margin:0;min-height:100vh}main{max-width:1100px;margin:0 auto;padding:32px 20px 48px}
.header{display:flex;align-items:center;gap:13px;padding-bottom:24px;margin-bottom:28px;border-bottom:1px solid #e1e7ef}
.mark{background:#2863bc;color:#fff;font-size:22px;font-weight:800;border-radius:13px;width:48px;height:48px;display:grid;place-items:center}
.eyebrow{color:#68829f;font-size:11px;letter-spacing:.12em;font-weight:800;text-transform:uppercase}
h1{font-size:25px;letter-spacing:-.03em;margin:2px 0 3px;color:#142a45}p{margin:0;color:#74869a;font-size:13px}
.layout{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(265px,1fr);gap:18px}
.panel{background:#fff;border:1px solid #e3e9f1;border-radius:15px;padding:26px;box-shadow:0 5px 22px rgba(39,68,103,.045)}
.panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:18px}
.tag{font-size:11px;color:#2874b9;font-weight:800;letter-spacing:.09em;text-transform:uppercase;background:#e9f4ff;padding:7px 11px;border-radius:99px}
#message{margin:4px 0 24px;font-size:18px;min-height:24px;font-weight:650;color:#18324c}
.meter-label{color:#506780;font-weight:700;font-size:13px;margin-bottom:4px}
.meter{background:#e9eff6;border-radius:12px;height:27px;overflow:hidden;margin:14px 0 11px;border:1px solid #dae5f0}
#fill{height:100%;width:0;background:linear-gradient(90deg,#288bc9,#23b8a8,#edbb56);transition:width .35s ease-out}
.row{display:flex;justify-content:space-between;color:#71859b;font-size:12px;gap:10px}#speech{font-size:13px;margin:16px 0 24px;color:#5c7187}
button{border:1px solid #d8e3f0;border-radius:9px;padding:12px 13px;font-size:13px;font-weight:750;background:#fff;color:#275681;cursor:pointer;transition:background .15s,border-color .15s}
button:hover:not(:disabled){background:#edf5ff;border-color:#9bbfe5}button.primary{background:#286bc3;border-color:#286bc3;color:#fff}button.primary:hover:not(:disabled){background:#1958ab}
button.danger{color:#b44444;background:#fff6f6;border-color:#f3d3d3}button.danger:hover:not(:disabled){background:#ffecec}
button:disabled{opacity:.43;cursor:not-allowed}.buttons{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:13px}
label{font-size:13px;color:#405b78;font-weight:750}
input{width:100%;margin-top:9px;border:1px solid #d5e0eb;background:#fafcff;color:#18324c;padding:13px;border-radius:9px;font-size:14px;outline:none}
input:focus{border-color:#3b81cc;box-shadow:0 0 0 3px #e3effc}input::placeholder{color:#a1afbe}
small{color:#8594a5;font-size:12px;line-height:1.8}h2{font-size:16px;letter-spacing:-.01em;margin:0;color:#1e3651}
ul{padding:0;list-style:none;margin:15px 0}li{background:#f5f8fc;border:1px solid #eaf0f6;border-radius:9px;padding:12px;margin-bottom:8px;display:flex;justify-content:space-between;gap:8px;color:#274767;font-size:13px;font-weight:600}
li .sample-count{color:#8293a6;font-size:12px;font-weight:500}
.delete-profile{padding:4px 7px;color:#b44444;border:0;background:transparent;font-size:18px;line-height:1}
.delete-profile:hover:not(:disabled){background:#ffecec;color:#a02020}
#results{margin-top:19px;white-space:pre-line;color:#215a94;line-height:1.6;font-size:14px;font-weight:650}
kbd{font-family:inherit;border:1px solid currentColor;border-radius:4px;padding:1px 5px;opacity:.75;margin-left:4px;font-size:11px}
footer{color:#91a0ae;font-size:11px;margin-top:23px;line-height:1.7}
@media(max-width:720px){.layout{grid-template-columns:1fr}main{padding:19px 12px}.panel{padding:20px}}
</style></head><body><main><header class="header"><div class="mark">R</div><div><div class="eyebrow">Robot Console / Temi / Audio</div><h1>Voice Recognition</h1><p>Local speaker enrollment and identification · SpeechBrain</p></div></header>
<p style="margin:-12px 0 18px"><a href="/" style="color:#286bc3;text-decoration:none;font-weight:700">← Home</a></p><div class="layout"><section class="panel"><div class="panel-head"><h2>Microphone &amp; recording</h2><span class="tag" id="mode">IDLE</span></div><div id="message">Ready</div><div class="meter-label">Live input level</div>
<div class="meter"><div id="fill"></div></div><div class="row"><span id="level">RMS 0.00000 · Peak 0.00000</span><span id="percent">−∞ dBFS</span></div>
<div id="speech">Waiting for microphone audio...</div>
<label for="name">Name for enrollment (required before recording)</label><input id="name" placeholder="e.g. John" autocomplete="off">
<div class="buttons"><button class="primary" id="startEnroll">▶ Start recording <kbd>S</kbd></button><button id="stopEnroll">■ Stop & save <kbd>E</kbd></button>
<button class="primary" id="startTest">◉ Test voice <kbd>T</kbd></button><button id="stopTest">■ Identify <kbd>R</kbd></button></div>
<div id="results"></div><button class="danger" id="quit" style="margin-top:22px">Quit server <kbd>Q</kbd></button></section>
<aside class="panel"><h2>Saved voice profiles</h2><ul id="people"><li>No profiles yet</li></ul><small>Profiles saved locally in voice_profiles.json</small>
<h2 style="margin-top:30px">Keys</h2><small>S start enrollment<br>E stop + save<br>T start voice test<br>R stop + identify<br>Q quit app</small></aside></div>
<footer>Voice activity indicator uses microphone volume (not a speech classifier). Cosine similarity is not a probability. Record only with permission.</footer></main>
<script>
let latest={mode:null,processing:false};
const el=id=>document.getElementById(id);
async function api(path,body={}){
 try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');await update();}
 catch(err){el('message').textContent='Error: '+err.message;}
}
async function update(){
 try{const r=await fetch('/status');if(!r.ok)return;let s=await r.json();latest=s;
 el('mode').textContent=s.processing?'ANALYZING':(s.mode?s.mode.toUpperCase():'IDLE');
 el('message').textContent=s.message;
 // Display-only meter: quiet audio stays low, -15 dBFS fills the bar.
 const db=20*Math.log10(Math.max(s.rms,1e-8));
 const pct=Math.round(100*Math.pow(Math.max(0,Math.min(1,(db+42)/27)),2)); el('fill').style.width=pct+'%';
 el('percent').textContent=(db<=-120?'−∞':db.toFixed(1))+' dBFS';
 el('level').textContent=`RMS ${s.rms.toFixed(5)} · Peak ${s.peak.toFixed(5)}`;
 el('speech').textContent=s.rms>0.01?'● Audio detected':s.rms>0.002?'◌ Quiet audio detected':'○ Silent / very quiet';
 el('startEnroll').disabled=!!s.mode||s.processing||!el('name').value.trim();el('name').disabled=!!s.mode||s.processing;el('startTest').disabled=!!s.mode||s.processing||!s.people.length;
 el('stopEnroll').disabled=s.mode!=='enroll'||s.processing;el('stopTest').disabled=s.mode!=='identify'||s.processing;
 el('people').innerHTML='';if(!s.people.length){const li=document.createElement('li');li.textContent='No profiles yet';el('people').appendChild(li);}
 s.people.forEach(p=>{
    const li=document.createElement('li');
    const name=document.createElement('span');name.textContent=p.name;
    const actions=document.createElement('span');actions.style.display='flex';actions.style.alignItems='center';actions.style.gap='8px';
    const count=document.createElement('span');count.className='sample-count';count.textContent=p.samples+' sample(s)';
    const del=document.createElement('button');del.className='delete-profile';del.textContent='🗑';del.title='Delete '+p.name;del.setAttribute('aria-label','Delete '+p.name);
    del.disabled=!!s.mode||s.processing;
    del.onclick=async()=>{if(confirm('Delete all voice samples for '+p.name+'?')) await api('/delete_profile',{name:p.name});};
    actions.append(count,del);li.append(name,actions);el('people').appendChild(li);
  });
 if(s.result){el('results').textContent=s.result.kind==='identify'?('RESULT: '+s.result.name+'\n'+s.result.scores.map(x=>x.name+': '+x.score).join('\n')):'Saved: '+s.result.name;}
 }catch(_e){}
}
el('startEnroll').onclick=()=>{
  const name=el('name').value.trim();
  if(!name){el('message').textContent='Enter a name before recording.';return;}
  el('results').textContent='';api('/start',{mode:'enroll',name});
};
el('stopEnroll').onclick=async()=>{
  try{
    const r=await fetch('/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const data=await r.json();
    if(!r.ok)throw Error(data.error||'Failed to save recording');
    if(data.result && data.result.kind==='enroll')el('name').value='';
    await update();
  }catch(err){el('message').textContent='Error: '+err.message;await update();}
};
el('startTest').onclick=()=>{el('results').textContent='';api('/start',{mode:'identify'});};
el('stopTest').onclick=()=>api('/stop');
el('name').addEventListener('input',()=>{if(!latest.mode&&!latest.processing)el('startEnroll').disabled=!el('name').value.trim();});
el('quit').onclick=async()=>{await api('/quit');document.body.innerHTML='<main><h1>Server stopped.</h1><p>Close this tab.</p></main>';};
document.addEventListener('keydown',e=>{if(e.ctrlKey||e.altKey||e.metaKey||e.repeat)return;if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;
const k=e.key.toLowerCase();const map={s:'startEnroll',e:'stopEnroll',t:'startTest',r:'stopTest',q:'quit'};
if(map[k]&&!el(map[k]).disabled){e.preventDefault();el(map[k]).click();}});
setInterval(update,150);update();
</script></body></html>'''


@app.get('/single')


def index():
    return render_template_string(PAGE)



HOME = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Console · Voice Lab</title><style>
:root{font-family:Inter,ui-sans-serif,system-ui,Arial;background:#f4f7fb;color:#1c3552}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center}main{width:min(780px,95vw)}h1{text-align:center;font-size:29px;margin:0 0 7px}p{text-align:center;color:#8495ab;margin:0 0 28px}.tiles{display:grid;grid-template-columns:1fr 1fr;gap:20px}.tile{height:205px;border-radius:19px;padding:26px;display:flex;flex-direction:column;justify-content:end;color:white;text-decoration:none;box-shadow:0 10px 30px #30476119;transition:transform .15s}.tile:not(.disabled):hover{transform:translateY(-4px)}.blue{background:linear-gradient(130deg,#266cc8,#123c80)}.red{background:linear-gradient(130deg,#e05a61,#961f33)}.disabled{background:#d3d9e1;color:#728092;box-shadow:none;cursor:not-allowed}.icon{font-size:36px;margin-bottom:auto}strong{font-size:21px}small{opacity:.8;margin-top:7px}@media(max-width:540px){.tiles{gap:12px}.tile{height:180px;padding:18px}strong{font-size:16px}}
</style></head><body><main><h1>Robot Console</h1><p>Select a voice experiment</p><div class="tiles">
<a class="tile blue" href="/single"><span class="icon">◉</span><strong>Single Speaker</strong><small>Enroll & identify saved voices</small></a>
<a class="tile red" href="/group"><span class="icon">◈</span><strong>Two Speakers</strong><small>SepFormer separation experiment</small></a>
<div class="tile disabled"><span class="icon">＋</span><strong>Coming soon</strong></div>
<div class="tile disabled"><span class="icon">＋</span><strong>Coming soon</strong></div></div></main></body></html>'''

GROUP = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Two Speaker Recognition</title><style>
:root{font-family:Inter,ui-sans-serif,system-ui,Arial;background:#f4f7fb;color:#20354e}*{box-sizing:border-box}main{max-width:910px;margin:36px auto;padding:0 16px}a{color:#ad3445;text-decoration:none;font-weight:700}.layout{display:grid;grid-template-columns:1.5fr 1fr;gap:18px}.panel{background:white;border:1px solid #e4e8ef;padding:25px;border-radius:16px;box-shadow:0 5px 25px #243b5310}h1{margin:15px 0 8px}p{color:#788aa2}button{cursor:pointer;border:0;border-radius:10px;padding:15px;font-size:15px;font-weight:700}button:disabled{opacity:.45;cursor:not-allowed}.start{background:#b52e48;color:white}.stop{background:#e9edf3;color:#243c58}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px}.meter{height:23px;background:#eef2f7;border-radius:11px;overflow:hidden;margin:21px 0 12px}.fill{height:100%;width:0;background:linear-gradient(90deg,#e8766a,#cc3f55);transition:width .35s}.small{font-size:12px;color:#8a9bb0}.result{background:#f5f7fb;border-radius:10px;padding:13px;margin:10px 0;font-weight:700}li{margin:10px 0}.status{min-height:50px;color:#324e69;font-weight:600}@media(max-width:700px){.layout{grid-template-columns:1fr}}
</style></head><body><main><a href="/">← Home</a><h1>Two Speaker Recognition</h1><p>Experimental · one microphone · two simultaneous voices</p><div class="layout"><section class="panel"><h2>Group recording</h2><div class="status" id="message">Ready.</div><div class="meter"><div id="fill" class="fill"></div></div><p id="db">−∞ dBFS</p><div class="controls"><button class="start" id="start">Start recording (S)</button><button class="stop" id="stop">Stop & identify (E)</button></div><div id="results"></div><section id="training" style="display:none;margin-top:22px"><h2>Train separated voice profiles</h2><p class="small">Listen first, then choose who is actually speaking in each channel. Do not save a channel with mixed or unclear voices.</p><div id="tracks"></div></section><p class="small">Two-speaker separation runs after stopping. CPU processing may be slow. It may output false positives when only one person speaks.</p></section>
<aside class="panel"><h2>Enrolled voices</h2><ul id="names"></ul><p class="small">Save normal profiles from the blue Single Speaker screen. Then record two people here, listen to each separated output, and save the correct identity.</p></aside></div></main><script>
const el=x=>document.getElementById(x);let state={},renderedGroup=null;
async function post(url,body={}){try{let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let j=await r.json();if(!r.ok)throw Error(j.error||'Request failed');await update()}catch(e){el('message').textContent='Error: '+e.message}}
async function update(){try{let r=await fetch('/status');let s=await r.json();state=s;el('message').textContent=s.message;let db=20*Math.log10(Math.max(s.rms,1e-8));let pct=Math.round(100*Math.pow(Math.max(0,Math.min(1,(db+42)/27)),2));el('fill').style.width=pct+'%';el('db').textContent=db.toFixed(1)+' dBFS';el('start').disabled=!!s.mode||s.processing||!s.people.length;el('stop').disabled=s.mode!=='group'||s.processing;el('names').replaceChildren(...s.people.map(p=>{let li=document.createElement('li');li.textContent=p.name+' ('+p.samples+' normal, '+p.separated_samples+' separated)';return li}));if(s.result?.kind==='group' && renderedGroup!==JSON.stringify(s.result)){
 renderedGroup=JSON.stringify(s.result);
 el('message').textContent='Estimated '+s.result.count+' speaker(s). '+s.result.note;
 el('results').replaceChildren(...s.result.speakers.map(p=>{let d=document.createElement('div');d.className='result';d.textContent='Voice '+p.channel+': '+p.name+(p.score===null?'':p.name==='Unknown speaker'?(p.closest_name?' (closest: '+p.closest_name+' · '+p.score+')':' (no reliable match · '+p.score+')'):' · similarity '+p.score)+(p.reference?' · '+p.reference:'');return d}));
 el('training').style.display='block';el('tracks').replaceChildren(...s.result.speakers.map(p=>{
  let d=document.createElement('div');d.className='result';
  let heading=document.createElement('div');heading.textContent='Voice '+p.channel;
  let audio=document.createElement('audio');audio.controls=true;audio.style.width='100%';audio.src='/group_audio/'+p.channel+'?v='+Date.now();
  let select=document.createElement('select');select.style.cssText='padding:10px;margin:8px 8px 8px 0;border-radius:8px;max-width:180px';
  let placeholder=document.createElement('option');placeholder.value='';placeholder.textContent='Choose person';select.appendChild(placeholder);
  s.people.forEach(person=>{let opt=document.createElement('option');opt.value=person.name;opt.textContent=person.name;select.appendChild(opt)});
  let save=document.createElement('button');save.textContent='Save separated profile';save.style.cssText='background:#b52e48;color:white;padding:11px';
  save.disabled=s.processing;
  save.onclick=async()=>{if(!select.value){el('message').textContent='Choose the person for Voice '+p.channel;return}save.disabled=true;await post('/save_separated',{channel:p.channel,name:select.value});save.disabled=false};
  d.append(heading,audio,select,save);return d}));
 }}catch(e){}}
el('start').onclick=()=>{renderedGroup=null;el('results').replaceChildren();el('training').style.display='none';el('tracks').replaceChildren();post('/start',{mode:'group'})};el('stop').onclick=()=>post('/stop');document.addEventListener('keydown',e=>{if(e.repeat||e.ctrlKey||e.altKey||e.metaKey)return;let k=e.key.toLowerCase();if(k==='s'&&!el('start').disabled)el('start').click();if(k==='e'&&!el('stop').disabled)el('stop').click()});setInterval(update,250);update();
</script></body></html>'''

@app.get('/')
def home():
    return render_template_string(HOME)

@app.get('/group')
def group_page():
    return render_template_string(GROUP)

def main():
    global model, stream, sample_rate
    print('Loading SpeechBrain model...')
    model = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir=str(ROOT / 'pretrained_models' / 'spkrec-ecapa-voxceleb'),
        run_opts={'device': 'cpu'},
        local_strategy=LocalStrategy.COPY,
    )
    # Use the default microphone rate; resample for SpeechBrain.
    device = sd.query_devices(kind='input')
    sample_rate = int(device['default_samplerate'])
    print(f"Microphone: {device['name']} @ {sample_rate} Hz")
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback)
    stream.start()
    with lock:
        state['message'] = 'Ready. Enter a name and click Start recording.'
    print('Open Voice Lab: \033]8;;http://127.0.0.1:5000\033\\http://127.0.0.1:5000\033]8;;\033\\')
    print('Press Ctrl+C in terminal to quit. (If the link is not clickable, copy the URL.)')
    webbrowser.open('http://127.0.0.1:5000')
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        stream.stop()
        stream.close()
if __name__ == '__main__':
    main()
