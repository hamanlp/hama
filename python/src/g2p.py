import struct
from pathlib import Path
import wasmtime

# Offsets in the result structure (all offsets are in bytes)
G2P_INPUT_OFFSET = 0
G2P_INPUT_BYTE_COUNT_OFFSET = 4
G2P_IPA_OFFSET = 8
G2P_IPA_BYTE_COUNT_OFFSET = 12

class Phonemizer:
    def __init__(self):
        self.wasm_instance = None
        self.memory = None
        self._init_phonemizer = None
        self._to_ipa = None
        self._allocUint8 = None
        self._deinit_result = None
        self._deinit_phonemizer = None
        self.phonemizer = None
        self.loaded = False
        self.store = None

    def load(self):
        # Create a Wasmtime store.
        self.store = wasmtime.Store()
        wasm_path = Path(__file__).parent / "hama-g2p.wasm"
        if not wasm_path.exists():
            raise FileNotFoundError(f"WASM file not found at {wasm_path}")
        # Open the file in binary mode and read its contents.
        wasm_bytes = wasm_path.read_bytes()
        # Create the module from the binary data.
        module = wasmtime.Module(self.store.engine, wasm_bytes)

        # Define the imported "jslog" function.
        def jslog_func(caller, x: int):
            print(x)
        jslog = wasmtime.Func(
            self.store,
            wasmtime.FuncType([wasmtime.ValType.i32()], []),
            jslog_func,
        )

        # Instantiate the module with the given imports.
        self.wasm_instance = wasmtime.Instance(self.store, module, [jslog])
        exports = self.wasm_instance.exports(self.store)

        # Retrieve exported functions.
        self._init_phonemizer = exports["init_phonemizer"]
        self._to_ipa = exports["to_ipa"]
        self._allocUint8 = exports["allocUint8"]
        self._deinit_result = exports["deinit_result"]
        self._deinit_phonemizer = exports["deinit_phonemizer"]
        self.memory = exports["memory"]

        self.loaded = True
        self.init_phonemizer()

    def init_phonemizer(self):
        if self._init_phonemizer is None:
            raise Exception("init_phonemizer function is not available")
        # Call the WASM function to initialize the phonemizer.
        self.phonemizer = self._init_phonemizer(self.store)

    def encode_string(self, string: str):
        if not self.loaded:
            raise Exception("WASM module not loaded")
        # Encode the string as UTF-8.
        encoded = string.encode("utf-8")
        byte_length = len(encoded)
        # Ask the WASM module to allocate memory for the string.
        pointer = self._allocUint8(self.store, byte_length)
        # Write the encoded bytes to the WASM memory at the given pointer.
        self.memory.write(self.store, pointer, encoded)
        return pointer, byte_length

    def to_ipa(self, text: str) -> str:
        if self._to_ipa is None:
            raise Exception("to_ipa function is not available")
        if self.phonemizer is None:
            raise Exception("phonemizer pointer is not available")
        
        # Encode the input string into WASM memory.
        input_ptr, input_len = self.encode_string(text)
        # Call the WASM function that performs G2P conversion.
        result_ptr = self._to_ipa(self.store, self.phonemizer, input_ptr, input_len)

        # Read the result structure from WASM memory.
        # The structure contains four unsigned 32-bit integers:
        # - input_address (at offset G2P_INPUT_OFFSET)
        # - input_byte_count (at offset G2P_INPUT_BYTE_COUNT_OFFSET)
        # - ipa_address (at offset G2P_IPA_OFFSET)
        # - ipa_byte_count (at offset G2P_IPA_BYTE_COUNT_OFFSET)
        input_address = struct.unpack(
            "<I", self.memory.read(self.store, result_ptr + G2P_INPUT_OFFSET, 4)
        )[0]
        input_byte_count = struct.unpack(
            "<I", self.memory.read(self.store, result_ptr + G2P_INPUT_BYTE_COUNT_OFFSET, 4)
        )[0]
        ipa_address = struct.unpack(
            "<I", self.memory.read(self.store, result_ptr + G2P_IPA_OFFSET, 4)
        )[0]
        ipa_byte_count = struct.unpack(
            "<I", self.memory.read(self.store, result_ptr + G2P_IPA_BYTE_COUNT_OFFSET, 4)
        )[0]

        # Decode the IPA result from WASM memory.
        ipa_bytes = self.memory.read(self.store, ipa_address, ipa_byte_count)
        ipa_str = ipa_bytes.decode("utf-8")

        # Clean up the result structure.
        self._deinit_result(self.store, self.phonemizer, result_ptr)
        return ipa_str

    def deinit(self):
        if self._deinit_phonemizer is not None and self.phonemizer is not None:
            self._deinit_phonemizer(self.store, self.phonemizer)

# Example usage:
if __name__ == "__main__":
    phonemizer = Phonemizer()
    phonemizer.load()
    input_text = "example"
    ipa_result = phonemizer.to_ipa(input_text)
    print("IPA:", ipa_result)
    phonemizer.deinit()
