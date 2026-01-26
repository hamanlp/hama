import type { BuildConfig } from "bun";
import dts from "bun-plugin-dts";

const defaultBuildConfig: BuildConfig = {
  entrypoints: ["./src/index.ts"],
  outdir: "./dist",
  minify: true,
  target: "browser",
};

await Bun.build({
  ...defaultBuildConfig,
  format: "esm",
  naming: "[dir]/[name].js",
  plugins: [dts()],
});

await Bun.build({
  ...defaultBuildConfig,
  format: "cjs",
  naming: "[dir]/[name].cjs",
});

await Bun.build({
  ...defaultBuildConfig,
  format: "iife",
  naming: "[dir]/[name].global.js",
  globalName: "Hama",
});
