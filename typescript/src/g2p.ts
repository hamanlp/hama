import * as ort from 'onnxruntime-web';

import { createRequire } from "module";
import onnxModel from '../../onnx/g2p.onnx';


const InferenceSession = ort.InferenceSession;
const Tensor = ort.Tensor;

class G2P {
  constructor() {
    //this.wasmFilePath = Bun.file("./hama.wasm");
  }

  async load(): Promise<void> {
    try {

      const session = await ort.InferenceSession.create(onnxModel);
      console.log(session)
    } catch (error) {
      console.error("Error loading G2P module:", error);
    }
  }

}

export {G2P};

