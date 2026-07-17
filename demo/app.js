// JS glue for the rusty-whisper wasm module (no bundler, no dependencies).
// Protocol: wasm_alloc -> copy bytes -> wasm_load_model / wasm_transcribe
// -> read JSON from wasm_result_ptr/len. Memory views must be recreated
// after any call that may grow wasm memory.

const status = (msg) => { document.getElementById("status").textContent = msg; };

let exports = null;
let modelLoaded = false;

async function init() {
  const { instance } = await WebAssembly.instantiateStreaming(
    fetch("rusty_whisper.wasm"), {}
  );
  exports = instance.exports;
  status("wasm ready — pick a model and an audio file");
  maybeEnable();
}

function copyIn(bytes) {
  const ptr = exports.wasm_alloc(bytes.byteLength);
  new Uint8Array(exports.memory.buffer, ptr, bytes.byteLength).set(bytes);
  return ptr;
}

function readResult() {
  const ptr = exports.wasm_result_ptr();
  const len = exports.wasm_result_len();
  const json = new TextDecoder().decode(new Uint8Array(exports.memory.buffer, ptr, len));
  return JSON.parse(json);
}

async function loadModel(file) {
  status(`loading model ${file.name} (${(file.size / 1e6).toFixed(0)} MB)…`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const ptr = copyIn(bytes);
  const rc = exports.wasm_load_model(ptr, bytes.byteLength);
  exports.wasm_free(ptr, bytes.byteLength);
  if (rc !== 0) {
    status(`model load failed: ${readResult().error}`);
    modelLoaded = false;
  } else {
    modelLoaded = true;
    status("model loaded");
  }
  maybeEnable();
}

// Decode any audio format to 16 kHz mono f32 via OfflineAudioContext.
async function decodeAudio(file) {
  const raw = await file.arrayBuffer();
  const probe = new AudioContext();
  const decoded = await probe.decodeAudioData(raw);
  probe.close();
  const frames = Math.ceil(decoded.duration * 16000);
  const off = new OfflineAudioContext(1, frames, 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

async function run() {
  const audioFile = document.getElementById("audio-file").files[0];
  const beam = parseInt(document.getElementById("beam").value, 10) || 5;
  document.getElementById("go").disabled = true;
  try {
    status("decoding audio…");
    const samples = await decodeAudio(audioFile);
    status(`transcribing ${(samples.length / 16000).toFixed(1)} s of audio… (single-threaded wasm — be patient)`);
    await new Promise((r) => setTimeout(r, 30)); // let the status paint
    const t0 = performance.now();
    const ptr = exports.wasm_alloc(samples.length * 4);
    new Float32Array(exports.memory.buffer, ptr, samples.length).set(samples);
    const rc = exports.wasm_transcribe(ptr, samples.length, beam);
    exports.wasm_free(ptr, samples.length * 4);
    if (rc !== 0) {
      status(`failed: ${readResult().error}`);
      return;
    }
    const result = readResult();
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    status(`done in ${secs} s — language: ${result.language}`);
    const list = document.getElementById("segments");
    list.innerHTML = "";
    for (const seg of result.segments) {
      const li = document.createElement("li");
      const time = document.createElement("time");
      time.textContent = `${seg.t0.toFixed(2)}–${seg.t1.toFixed(2)}s`;
      li.appendChild(time);
      li.appendChild(document.createTextNode(seg.text));
      list.appendChild(li);
    }
  } finally {
    document.getElementById("go").disabled = !(modelLoaded && document.getElementById("audio-file").files[0]);
  }
}

function maybeEnable() {
  const ready = exports && modelLoaded && document.getElementById("audio-file").files[0];
  document.getElementById("go").disabled = !ready;
}

document.getElementById("model-file").addEventListener("change", (e) => {
  if (e.target.files[0]) loadModel(e.target.files[0]);
});
document.getElementById("audio-file").addEventListener("change", maybeEnable);
document.getElementById("go").addEventListener("click", run);

init().catch((e) => status(`wasm init failed: ${e}`));
