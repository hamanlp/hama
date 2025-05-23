import type { BuildConfig } from "bun";
//import dts from "bun-plugin-dts";
import { cp } from "fs/promises";

const defaultBuildConfig: BuildConfig = {
  entrypoints: ["./src/index.ts"],
  outdir: "./dist",
  minify: true,
};

await Promise.all([
  Bun.build({
    ...defaultBuildConfig,
    //plugins: [dts()],
    format: "esm",
    naming: "[dir]/[name].js",
  }),
  Bun.build({
    ...defaultBuildConfig,
    format: "cjs",
    naming: "[dir]/[name].cjs",
  }),
]);
