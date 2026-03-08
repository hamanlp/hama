import { ASRNodeModel } from "../src/asr";
import { spawnSync } from "node:child_process";

const parseArg = (name: string, defaultValue?: string): string | undefined => {
  const idx = process.argv.indexOf(name);
  if (idx < 0) return defaultValue;
  return process.argv[idx + 1] ?? defaultValue;
};

const parseNum = (name: string, defaultValue: number): number => {
  const raw = parseArg(name);
  if (!raw) return defaultValue;
  const v = Number(raw);
  return Number.isFinite(v) ? v : defaultValue;
};

const hasFlag = (name: string): boolean => process.argv.includes(name);

const commandExists = (cmd: string): boolean => {
  const checker = process.platform === "win32" ? "where" : "command";
  const args = process.platform === "win32" ? [cmd] : ["-v", cmd];
  const result = spawnSync(checker, args, { stdio: "ignore", shell: process.platform !== "win32" });
  return result.status === 0;
};

const installHintForProgram = (program: string): string => {
  if (process.platform === "darwin") {
    return `Install '${program}' via Homebrew: brew install sox`;
  }
  if (process.platform === "linux") {
    if (program === "arecord") {
      return "Install ALSA tools: sudo apt-get install -y alsa-utils";
    }
    return "Install SoX: sudo apt-get install -y sox";
  }
  return `Install '${program}' and make sure it is available in PATH.`;
};

const pcm16ToFloat32 = (buf: Buffer): Float32Array => {
  const n = Math.floor(buf.length / 2);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = buf.readInt16LE(i * 2) / 32768;
  }
  return out;
};

const requiredSilenceMs = (utteranceMs: number): number => {
  if (utteranceMs < 3000) return 1000;
  if (utteranceMs < 5000) {
    const ratio = (utteranceMs - 3000) / 2000;
    return Math.round(1000 + ratio * (500 - 1000));
  }
  if (utteranceMs < 12000) {
    const ratio = (utteranceMs - 5000) / 7000;
    return Math.round(500 + ratio * (200 - 500));
  }
  if (utteranceMs <= 17000) {
    const ratio = (utteranceMs - 12000) / 5000;
    return Math.round(200 + ratio * (100 - 200));
  }
  return 0;
};

const main = async (): Promise<void> => {
  const sampleRate = parseNum("--sample-rate", 16000);
  const vadThreshold = parseNum("--vad-threshold", 0.6);
  const preSpeechPadMs = parseNum("--vad-speech-pad-ms", 30);
  const minSpeechMs = parseNum("--min-utterance-ms", 180);
  const unkBias = parseNum("--unk-bias", -1.5);
  const showUnk = hasFlag("--show-unk");
  const inputDevice = parseArg("--input-device");
  const modelPath = parseArg("--model");
  const recordProgram =
    parseArg("--record-program") ??
    (process.platform === "linux" ? "arecord" : "sox");

  let NonRealTimeVADCtor: any;
  let ResamplerCtor: any;
  let MessageEnum: any;
  let recordMod: any;
  try {
    ({ NonRealTimeVAD: NonRealTimeVADCtor, Resampler: ResamplerCtor, Message: MessageEnum } = await import("@ricky0123/vad-node"));
  } catch (_err) {
    throw new Error(
      "Missing optional dependency '@ricky0123/vad-node'. Install with: bun add -d @ricky0123/vad-node",
    );
  }
  try {
    recordMod = await import("node-record-lpcm16");
  } catch (_err) {
    throw new Error(
      "Missing optional dependency 'node-record-lpcm16'. Install with: bun add -d node-record-lpcm16",
    );
  }

  if (sampleRate < 16000) {
    throw new Error("This TS live example requires --sample-rate >= 16000.");
  }

  const asr = await ASRNodeModel.create({
    modelPath,
    blankBias: -0.1,
    unkBias,
  });

  const frameSamples = 512;
  const targetVadSampleRate = 16000;
  const frameMs = (frameSamples * 1000) / targetVadSampleRate;
  const preSpeechPadFrames = Math.max(1, Math.round(preSpeechPadMs / frameMs));
  const minSpeechFrames = Math.max(1, Math.ceil(minSpeechMs / frameMs));
  const vad = await NonRealTimeVADCtor.new({
    positiveSpeechThreshold: vadThreshold,
    negativeSpeechThreshold: Math.max(0, vadThreshold - 0.15),
    frameSamples,
    preSpeechPadFrames,
    minSpeechFrames,
    redemptionFrames: Math.ceil(1000 / frameMs),
  });
  const frameProcessor = vad.frameProcessor;
  if (!frameProcessor) {
    throw new Error("Failed to initialize frame-level VAD processor.");
  }
  const resampler = new ResamplerCtor({
    nativeSampleRate: sampleRate,
    targetSampleRate: targetVadSampleRate,
    targetFrameSize: frameSamples,
  });

  if (!commandExists(recordProgram)) {
    throw new Error(
      `Missing recorder binary '${recordProgram}'. ${installHintForProgram(recordProgram)}`,
    );
  }

  let source: any;
  try {
    source = recordMod.record({
      sampleRateHertz: sampleRate,
      channels: 1,
      threshold: 0,
      audioType: "raw",
      verbose: false,
      device: inputDevice,
      recordProgram,
    });
  } catch (err: any) {
    const msg = String(err?.message ?? err ?? "");
    if (msg.includes("Executable not found in $PATH")) {
      throw new Error(
        `Recorder '${recordProgram}' is not available in PATH. ${installHintForProgram(recordProgram)}`,
      );
    }
    throw err;
  }
  const stream = source.stream();

  console.log("[live-ts] starting mic stream (Ctrl+C to stop)");
  console.log(`[live-ts] recorder=${recordProgram}`);
  console.log(
    "[live-ts] VAD: Silero threshold=0.6, silence=1000ms(<3s) -> 500ms(5s) -> 200ms(12s) -> 100ms(17s) -> 0ms(>17s)",
  );
  console.log("[live-ts] waiting for speech...");

  const chunkQueue: Buffer[] = [];
  let processing = false;
  let frameIndex = 0;
  let segmentStartFrame = -1;

  const pumpAudio = async (): Promise<void> => {
    if (processing) return;
    processing = true;
    try {
      while (chunkQueue.length > 0) {
        const chunk = chunkQueue.shift();
        if (!chunk) continue;
        const pcm = pcm16ToFloat32(chunk);
        const frames = resampler.process(pcm);
        for (const frame of frames) {
          if (segmentStartFrame >= 0) {
            const utteranceMs = (frameIndex - segmentStartFrame + 1) * frameMs;
            frameProcessor.options.redemptionFrames = Math.ceil(requiredSilenceMs(utteranceMs) / frameMs);
          } else {
            frameProcessor.options.redemptionFrames = Math.ceil(1000 / frameMs);
          }

          const { msg, audio } = await frameProcessor.process(frame);
          if (msg === MessageEnum.SpeechStart) {
            segmentStartFrame = Math.max(0, frameIndex - preSpeechPadFrames);
            const startMs = segmentStartFrame * frameMs;
            console.log(`[vad-ts] speech start @ ${Math.round(startMs)}ms`);
          } else if (msg === MessageEnum.SpeechEnd && audio) {
            const startMs = Math.max(0, segmentStartFrame) * frameMs;
            const endMs = (frameIndex + 1) * frameMs;
            segmentStartFrame = -1;
            console.log(`[vad-ts] speech ${Math.round(startMs)}ms -> ${Math.round(endMs)}ms`);

            const result = await asr.transcribeWaveform(audio, targetVadSampleRate);
            const tokens = showUnk ? result.phonemes : result.phonemes.filter((t) => t !== "<unk>");
            const text = tokens.join(" ").trim();
            if (text.length > 0) console.log(`[phonemes-ts] ${text}`);
          }
          frameIndex += 1;
        }
      }
    } catch (err: any) {
      console.error("[live-ts] processing error:", err?.message ?? err);
    } finally {
      processing = false;
      if (chunkQueue.length > 0) {
        void pumpAudio();
      }
    }
  };

  stream.on("data", (chunk: Buffer) => {
    chunkQueue.push(chunk);
    void pumpAudio();
  });
  stream.on("error", (err: Error) => {
    console.error("[audio] stream error:", err.message);
  });

  const stop = async () => {
    try {
      const final = frameProcessor.endSegment();
      if (final.msg === MessageEnum.SpeechEnd && final.audio) {
        const result = await asr.transcribeWaveform(final.audio, targetVadSampleRate);
        const tokens = showUnk ? result.phonemes : result.phonemes.filter((t) => t !== "<unk>");
        const text = tokens.join(" ").trim();
        if (text.length > 0) console.log(`[phonemes-ts] ${text}`);
      }
    } catch {
      // ignore
    }
    try {
      source.stop();
    } catch {
      // ignore
    }
    console.log("\n[live-ts] stopping...");
    process.exit(0);
  };
  process.on("SIGINT", () => { void stop(); });
  process.on("SIGTERM", () => { void stop(); });
};

main().catch((err) => {
  console.error("[live-ts] failed");
  console.error(err);
  process.exit(1);
});
