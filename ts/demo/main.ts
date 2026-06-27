// In-browser playground for the hama engine: G2P (text→phonemes),
// ASR (audio→phonemes), and P2G (phonemes→text). Uses the real published
// browser API running the WASM engine.
import { ASRBrowserModel, G2PBrowserModel, P2GBrowserModel } from "../src/browser";

const $ = (id: string) => document.getElementById(id) as HTMLElement;
const fmt = (ms: number, n: number, unit: string) =>
  `${n} ${unit} · ${ms.toFixed(0)} ms · ${Math.round(n / (ms / 1000))} ${unit}/s`;

// Lazy model loaders — weights fetch only when a tab is first used.
function lazy<T>(create: () => Promise<T>, stat: HTMLElement, label: string, mb: string) {
  let p: Promise<T> | null = null;
  return () => {
    if (!p) {
      const t0 = performance.now();
      stat.textContent = `Loading ${label} model (${mb}, first time only)…`;
      p = create().then((m) => {
        stat.textContent = `${label} ready in ${Math.round(performance.now() - t0)} ms.`;
        return m;
      });
    }
    return p;
  };
}

// ───────────── Tabs ─────────────
const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>(".tab"));
const views = Array.from(document.querySelectorAll<HTMLElement>(".view"));
function activateTab(name: string) {
  for (const t of tabs) t.classList.toggle("active", t.dataset.tab === name);
  for (const v of views) v.classList.toggle("active", v.dataset.view === name);
}
for (const tab of tabs) tab.addEventListener("click", () => activateTab(tab.dataset.tab!));

// Cross-tab hook: the ASR tab hands its word-segmented phonemes to P2G.
let p2gRun: ((text: string) => void) | null = null;

// ───────────── G2P ─────────────
{
  const inp = $("g2p-in") as HTMLTextAreaElement;
  const out = $("g2p-out");
  const display = $("g2p-display");
  const stat = $("g2p-stat");
  const go = $("g2p-go") as HTMLButtonElement;
  const getModel = lazy(() => G2PBrowserModel.create(), stat, "G2P", "~4 MB");
  go.disabled = false;

  const PRESETS = ["Mister Quilter is the apostle of the middle classes.", "hello world", "안녕하세요", "Dr. Park's 3 cats"];
  const box = $("g2p-presets");
  for (const text of PRESETS) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = text.length > 28 ? text.slice(0, 27) + "…" : text;
    b.title = text;
    b.addEventListener("click", () => { inp.value = text; run(); });
    box.appendChild(b);
  }

  async function run() {
    const text = inp.value.trim();
    if (!text) return;
    go.disabled = true;
    out.textContent = "…"; display.textContent = "";
    try {
      const m = await getModel();
      const t0 = performance.now();
      const r = await m.predict(text, { preserveLiterals: "punct" });
      const ms = performance.now() - t0;
      out.textContent = r.ipa || "(empty)";
      display.innerHTML = `<b>display:</b> ${r.displayIpa || "—"} &nbsp;·&nbsp; ${fmt(ms, r.alignments.length, "phon")}`;
    } catch (e) {
      out.textContent = "Error: " + (e as Error).message;
    } finally {
      go.disabled = false;
    }
  }
  go.addEventListener("click", run);
  inp.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); });
}

// ───────────── ASR ─────────────
{
  const fileInput = $("asr-file") as HTMLInputElement;
  const audio = $("asr-audio") as HTMLAudioElement;
  const recBtn = $("asr-rec") as HTMLButtonElement;
  const go = $("asr-go") as HTMLButtonElement;
  const out = $("asr-out");
  const words = $("asr-words");
  const stat = $("asr-stat");
  const send = $("asr-send") as HTMLButtonElement;
  const getModel = lazy(() => ASRBrowserModel.create(), stat, "ASR", "~5 MB");

  let pending: { data: Float32Array; sampleRate: number } | null = null;
  let lastWords = "";
  // Hand the word-segmented phonemes (with `|` from the model's <wb> tokens) to P2G.
  send.addEventListener("click", () => {
    if (!lastWords) return;
    activateTab("p2g");
    p2gRun?.(lastWords);
  });

  async function decodeBuffer(buf: ArrayBuffer) {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new Ctx();
    const audioBuf = await ctx.decodeAudioData(buf);
    await ctx.close();
    return { data: new Float32Array(audioBuf.getChannelData(0)), sampleRate: audioBuf.sampleRate };
  }

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    audio.src = URL.createObjectURL(file); audio.hidden = false;
    stat.textContent = "Decoding audio…";
    pending = await decodeBuffer(await file.arrayBuffer());
    stat.textContent = `Loaded ${(pending.data.length / pending.sampleRate).toFixed(1)} s @ ${pending.sampleRate} Hz — ready to transcribe.`;
    go.disabled = false;
  });

  // Mic recording -> webm/opus blob -> decodeAudioData -> waveform.
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
        recBtn.textContent = "● Record"; recBtn.classList.add("rec");
        const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
        audio.src = URL.createObjectURL(blob); audio.hidden = false;
        stat.textContent = "Decoding recording…";
        pending = await decodeBuffer(await blob.arrayBuffer());
        stat.textContent = `Recorded ${(pending.data.length / pending.sampleRate).toFixed(1)} s — ready to transcribe.`;
        go.disabled = false;
      };
      recorder.start();
      recBtn.textContent = "■ Stop"; recBtn.classList.remove("rec");
      stat.textContent = "Recording… click stop when done.";
    } catch (e) {
      stat.textContent = "Mic error: " + (e as Error).message;
    }
  });

  go.addEventListener("click", async () => {
    if (!pending) return;
    go.disabled = true;
    out.textContent = "…"; words.textContent = "";
    try {
      const m = await getModel();
      const t0 = performance.now();
      const r = await m.transcribeWaveform(pending.data, pending.sampleRate);
      const ms = performance.now() - t0;
      out.textContent = r.phonemeText || "(silence)";
      words.innerHTML = `<b>words:</b> ${r.wordPhonemeText || "—"} &nbsp;·&nbsp; ${r.numFrames} frames · ${ms.toFixed(0)} ms`;
      lastWords = r.wordPhonemeText;
      send.hidden = !lastWords;
    } catch (e) {
      out.textContent = "Error: " + (e as Error).message;
    } finally {
      go.disabled = false;
    }
  });
}

// ───────────── P2G ─────────────
{
  const input = $("phon") as HTMLTextAreaElement;
  const out = $("p2g-out");
  const tokensEl = $("p2g-tokens");
  const stat = $("p2g-stat");
  const go = $("p2g-go") as HTMLButtonElement;
  const getModel = lazy(() => P2GBrowserModel.create(), stat, "P2G", "~29 MB");
  go.disabled = false;

  const PRESETS = [
    { label: "i harry kilter…", phonemes: "aɪ | h ɛ ɹ i | k ɪ l t ɝ | æ m | eɪ" },
    { label: "he disn't worket all", phonemes: "h i | d ɪ s n ə t | w ɝ k ɪ t | ɔ l" },
    { label: "mister quilter…", phonemes: "m ɪ s t ɝ | k w ɪ l t ɝ | ɪ z | ð i | ə p ɑ s ə l" },
  ];
  const box = $("p2g-presets");
  for (const p of PRESETS) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = p.label; b.title = p.phonemes;
    b.addEventListener("click", () => { input.value = p.phonemes; run(); });
    box.appendChild(b);
  }

  async function run() {
    const phonemes = input.value.trim();
    if (!phonemes) return;
    go.disabled = true;
    out.textContent = "…"; tokensEl.textContent = "";
    try {
      const m = await getModel();
      const t0 = performance.now();
      const r = m.predict(phonemes);
      const ms = performance.now() - t0;
      out.textContent = r.text || "(empty)";
      tokensEl.textContent = r.tokens.join(" ");
      stat.textContent = fmt(ms, r.tokens.length, "tok");
    } catch (e) {
      out.textContent = "Error: " + (e as Error).message;
    } finally {
      go.disabled = false;
    }
  }
  go.addEventListener("click", run);
  input.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); });
  // Allow the ASR tab to push phonemes here and run them.
  p2gRun = (text: string) => { input.value = text; run(); };
}
