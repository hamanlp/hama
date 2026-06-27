// Single-flow demo: mic (or file) -> ASR (audio→phonemes) -> P2G (phonemes→text),
// showing the phonemes and the decoded text together. Uses the real published
// browser API on the WASM engine.
import { ASRBrowserModel, P2GBrowserModel } from "../src/browser";

const $ = (id: string) => document.getElementById(id) as HTMLElement;
const recBtn = $("rec") as HTMLButtonElement;
const fileInput = $("file") as HTMLInputElement;
const audio = $("audio") as HTMLAudioElement;
const stat = $("stat");
const phonEl = $("phon");
const wordsEl = $("words");
const textEl = $("text");
const asrMs = $("asr-ms");
const p2gMs = $("p2g-ms");

// Warm both models on load so the first recording runs immediately.
let modelsPromise: Promise<[ASRBrowserModel, P2GBrowserModel]> | null = null;
function getModels() {
  if (!modelsPromise) {
    const t0 = performance.now();
    modelsPromise = Promise.all([ASRBrowserModel.create(), P2GBrowserModel.create()]).then((m) => {
      stat.textContent = `Models ready in ${Math.round(performance.now() - t0)} ms — record or upload a clip.`;
      return m;
    });
  }
  return modelsPromise;
}
getModels();

async function decodeBuffer(buf: ArrayBuffer): Promise<{ data: Float32Array; sampleRate: number }> {
  const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(buf);
  await ctx.close();
  return { data: new Float32Array(audioBuf.getChannelData(0)), sampleRate: audioBuf.sampleRate };
}

const renderPhonemes = (wordPhonemeText: string): string =>
  wordPhonemeText
    ? wordPhonemeText.replace(/\s\|\s/g, ' <span class="wb">|</span> ')
    : "(no speech detected)";

async function run(wave: Float32Array, sampleRate: number): Promise<void> {
  phonEl.textContent = "…"; textEl.textContent = "…"; wordsEl.textContent = "";
  asrMs.textContent = ""; p2gMs.textContent = "";
  try {
    const [asr, p2g] = await getModels();

    const a0 = performance.now();
    const asrOut = await asr.transcribeWaveform(wave, sampleRate);
    asrMs.textContent = `${(performance.now() - a0).toFixed(0)} ms · ${asrOut.numFrames} frames`;
    phonEl.innerHTML = renderPhonemes(asrOut.wordPhonemeText);
    const wordCount = asrOut.wordPhonemeText ? asrOut.wordPhonemeText.split(" | ").length : 0;
    wordsEl.textContent = wordCount ? `${wordCount} word${wordCount > 1 ? "s" : ""}` : "";

    if (!asrOut.wordPhonemeText) { textEl.textContent = "—"; return; }

    const p0 = performance.now();
    const p2gOut = p2g.predict(asrOut.wordPhonemeText);
    p2gMs.textContent = `${(performance.now() - p0).toFixed(0)} ms · ${p2gOut.tokens.length} tok`;
    textEl.textContent = p2gOut.text || "(empty)";
  } catch (e) {
    stat.textContent = "Error: " + (e as Error).message;
  }
}

// ── File upload ──
fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  audio.src = URL.createObjectURL(file); audio.hidden = false;
  stat.textContent = "Decoding audio…";
  const { data, sampleRate } = await decodeBuffer(await file.arrayBuffer());
  stat.textContent = `${(data.length / sampleRate).toFixed(1)} s @ ${sampleRate} Hz — transcribing…`;
  await run(data, sampleRate);
  stat.textContent = "Done — record or upload another clip.";
});

// ── Mic record ──
let recorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
recBtn.addEventListener("click", async () => {
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      recBtn.textContent = "● Record"; recBtn.classList.remove("recording");
      const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
      audio.src = URL.createObjectURL(blob); audio.hidden = false;
      stat.textContent = "Transcribing…";
      const { data, sampleRate } = await decodeBuffer(await blob.arrayBuffer());
      await run(data, sampleRate);
      stat.textContent = "Done — record again or upload a clip.";
    };
    recorder.start();
    recBtn.textContent = "■ Stop"; recBtn.classList.add("recording");
    stat.textContent = "Recording… click to stop.";
  } catch (e) {
    stat.textContent = "Mic error: " + (e as Error).message;
  }
});
