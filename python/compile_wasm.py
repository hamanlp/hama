from wasmer import engine, Store, Module
from pathlib import Path

# Locate the original WASM file.
wasm_file_path = Path(__file__).parent / "hama-g2p.wasm"
wasm_bytes = wasm_file_path.read_bytes()

jit_engine = engine.JIT(compiler="cranelift")
store = Store(jit_engine)
# This step must succeed in an environment that supports runtime compilation.
module = Module(store, wasm_bytes)

# Serialize the compiled module.
serialized_module = module.serialize()

# Save the serialized module.
precompiled_path = Path(__file__).parent / "hama-g2p-precompiled.bin"
precompiled_path.write_bytes(serialized_module)
print("Precompilation successful!")

