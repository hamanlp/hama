import enum
import struct
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import wasmer

# Offsets that mirror the layout in the Zig structs.
DISASSEMBLE_INPUT_OFFSET = 0
DISASSEMBLE_INPUT_BYTE_COUNT_OFFSET = 4
DISASSEMBLE_IS_HANGULS_OFFSET = 8
DISASSEMBLE_JAMOS_OFFSET = 12
DISASSEMBLE_JAMOS_COUNT_OFFSET = 16
DISASSEMBLE_JAMOS_BYTE_COUNT_OFFSET = 20
DISASSEMBLE_SYLLABLE_POSITIONS_OFFSET = 24

ASSEMBLE_INPUT_OFFSET = 0
ASSEMBLE_INPUT_BYTE_COUNT_OFFSET = 4
ASSEMBLE_CHARACTERS_OFFSET = 8
ASSEMBLE_CHARACTERS_COUNT_OFFSET = 12
ASSEMBLE_CHARACTERS_BYTE_COUNT_OFFSET = 16


class SyllablePosition(enum.Enum):
    CODA = 0
    NUCLEUS = 1
    ONSET = 2
    NOT_APPLICABLE = 3


class DisassembleResult(TypedDict):
    input: str
    is_hanguls: List[bool]
    text: str
    syllable_positions: List[SyllablePosition]


class AssembleResult(TypedDict):
    input: str
    text: str


def py_log(value: int) -> None:
    """Simple logging hook for the WASM module."""
    print(f"WASM log: {value}")


class JamoParser:
    def __init__(self) -> None:
        self.wasm_file_path = Path(__file__).parent / "hama-g2p.wasm"
        if not self.wasm_file_path.exists():
            raise FileNotFoundError(f"WASM file not found at: {self.wasm_file_path}")

        self.store: Optional[wasmer.Store] = None
        self.instance: Optional[wasmer.Instance] = None
        self.memory: Optional[wasmer.Memory] = None

        self._disassemble = None
        self._cleanup_disassemble = None
        self._assemble = None
        self._cleanup_assemble = None
        self._alloc_uint8 = None

        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return

        try:
            self.store = wasmer.Store()
            module = wasmer.Module(self.store, self.wasm_file_path.read_bytes())
            import_object = {
                "env": {
                    "jslog": wasmer.Function(self.store, py_log),
                }
            }
            self.instance = wasmer.Instance(module, import_object)
            self.memory = self.instance.exports.memory

            exports = self.instance.exports
            self._disassemble = getattr(exports, "disassemble", None)
            self._cleanup_disassemble = getattr(exports, "cleanup_disassemble", None)
            self._assemble = getattr(exports, "assemble", None)
            self._cleanup_assemble = getattr(exports, "cleanup_assemble", None)
            self._alloc_uint8 = getattr(exports, "allocUint8", None)

            required = {
                "memory": self.memory,
                "disassemble": self._disassemble,
                "cleanup_disassemble": self._cleanup_disassemble,
                "assemble": self._assemble,
                "cleanup_assemble": self._cleanup_assemble,
                "allocUint8": self._alloc_uint8,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise RuntimeError(
                    "One or more required exports not found in WASM module: "
                    + ", ".join(missing)
                )

            self.loaded = True
        except Exception:
            self._reset_state()
            raise

    def _reset_state(self) -> None:
        self.store = None
        self.instance = None
        self.memory = None
        self._disassemble = None
        self._cleanup_disassemble = None
        self._assemble = None
        self._cleanup_assemble = None
        self._alloc_uint8 = None
        self.loaded = False

    def _check_loaded(self) -> None:
        if not self.loaded or self.instance is None or self.memory is None:
            raise RuntimeError("WASM module is not loaded. Call load() first.")
        if self._alloc_uint8 is None:
            raise RuntimeError("allocUint8 export is not available.")

    def _encode_string(self, text: str) -> Tuple[int, int]:
        self._check_loaded()
        encoded = text.encode("utf-8")
        pointer = self._alloc_uint8(len(encoded))
        if not isinstance(pointer, int) or pointer == 0:
            raise MemoryError("Failed to allocate memory in WASM.")

        memory_view = self.memory.uint8_view(offset=pointer)  # type: ignore[attr-defined]
        if len(memory_view) < len(encoded):
            raise MemoryError(
                f"Allocated WASM memory ({len(memory_view)} bytes) is smaller than required ({len(encoded)} bytes)."
            )
        memory_view[0 : len(encoded)] = encoded
        return pointer, len(encoded)

    def _decode_string(self, pointer: int, length: int) -> str:
        self._check_loaded()
        if length == 0:
            return ""
        memory_view = self.memory.uint8_view(offset=pointer)  # type: ignore[attr-defined]
        if len(memory_view) < length:
            raise MemoryError(
                f"Attempting to read {length} bytes from WASM memory, but view only has {len(memory_view)} bytes starting at offset {pointer}."
            )
        return bytes(memory_view[0:length]).decode("utf-8")

    def _read_uint32(self, address: int) -> int:
        self._check_loaded()
        try:
            buffer = self.memory.buffer  # type: ignore[attr-defined]
            return struct.unpack_from("<I", buffer, address)[0]
        except struct.error as exc:
            raise MemoryError(
                f"Failed to read uint32 at address {address}: {exc}"
            ) from exc

    def _read_uint8_array(self, address: int, count: int) -> bytes:
        self._check_loaded()
        if count == 0:
            return b""
        memory_view = self.memory.uint8_view(offset=address)  # type: ignore[attr-defined]
        if len(memory_view) < count:
            raise MemoryError(
                f"Attempting to read {count} bytes from WASM memory, but view only has {len(memory_view)} bytes starting at offset {address}."
            )
        return bytes(memory_view[0:count])

    def disassemble(self, text: str) -> DisassembleResult:
        self._check_loaded()
        if self._disassemble is None or self._cleanup_disassemble is None:
            raise RuntimeError("disassemble exports are not available.")

        input_ptr, input_len = self._encode_string(text)
        result_ptr = 0
        try:
            result_ptr = self._disassemble(input_ptr, input_len, True)
            if not isinstance(result_ptr, int) or result_ptr == 0:
                raise RuntimeError("WASM disassemble returned an invalid pointer.")

            input_address = self._read_uint32(result_ptr + DISASSEMBLE_INPUT_OFFSET)
            input_byte_count = self._read_uint32(
                result_ptr + DISASSEMBLE_INPUT_BYTE_COUNT_OFFSET
            )
            is_hanguls_address = self._read_uint32(
                result_ptr + DISASSEMBLE_IS_HANGULS_OFFSET
            )
            jamos_address = self._read_uint32(result_ptr + DISASSEMBLE_JAMOS_OFFSET)
            jamos_count = self._read_uint32(
                result_ptr + DISASSEMBLE_JAMOS_COUNT_OFFSET
            )
            jamos_byte_count = self._read_uint32(
                result_ptr + DISASSEMBLE_JAMOS_BYTE_COUNT_OFFSET
            )
            syllable_positions_address = self._read_uint32(
                result_ptr + DISASSEMBLE_SYLLABLE_POSITIONS_OFFSET
            )

            original_input = self._decode_string(input_address, input_byte_count)
            is_hanguls_raw = self._read_uint8_array(is_hanguls_address, jamos_count)
            jamos = self._decode_string(jamos_address, jamos_byte_count)
            syllable_positions_raw = self._read_uint8_array(
                syllable_positions_address, jamos_count
            )

            syllable_positions: List[SyllablePosition] = []
            for value in syllable_positions_raw:
                try:
                    syllable_positions.append(SyllablePosition(value))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid syllable position value: {value}"
                    ) from exc

            return {
                "input": original_input,
                "text": jamos,
                "is_hanguls": [bool(v) for v in is_hanguls_raw],
                "syllable_positions": syllable_positions,
            }
        finally:
            if result_ptr and self._cleanup_disassemble:
                self._cleanup_disassemble(result_ptr)

    def assemble(self, text: str) -> AssembleResult:
        self._check_loaded()
        if self._assemble is None or self._cleanup_assemble is None:
            raise RuntimeError("assemble exports are not available.")

        input_ptr, input_len = self._encode_string(text)
        result_ptr = 0
        try:
            result_ptr = self._assemble(input_ptr, input_len)
            if not isinstance(result_ptr, int) or result_ptr == 0:
                raise RuntimeError("WASM assemble returned an invalid pointer.")

            input_address = self._read_uint32(result_ptr + ASSEMBLE_INPUT_OFFSET)
            input_byte_count = self._read_uint32(
                result_ptr + ASSEMBLE_INPUT_BYTE_COUNT_OFFSET
            )
            characters_address = self._read_uint32(result_ptr + ASSEMBLE_CHARACTERS_OFFSET)
            characters_byte_count = self._read_uint32(
                result_ptr + ASSEMBLE_CHARACTERS_BYTE_COUNT_OFFSET
            )

            return {
                "input": self._decode_string(input_address, input_byte_count),
                "text": self._decode_string(characters_address, characters_byte_count),
            }
        finally:
            if result_ptr and self._cleanup_assemble:
                self._cleanup_assemble(result_ptr)

    def encode_string(self, text: str) -> Tuple[int, int]:
        """Expose the raw encoder for callers that want direct buffer access."""
        return self._encode_string(text)

    def call_wasm_function(self, func_name: str, *args):
        """Invoke an arbitrary exported WASM function."""
        self._check_loaded()
        if self.instance is None:
            raise RuntimeError("WASM module is not instantiated.")
        func = getattr(self.instance.exports, func_name, None)
        if not callable(func):
            raise RuntimeError(f"{func_name} is not a callable export in the WASM module.")
        return func(*args)


if __name__ == "__main__":
    parser = JamoParser()
    parser.load()
    sample = "안녕하세요"
    print("Disassemble:", parser.disassemble(sample))
    assembled = parser.assemble("ㄱㅗㄱㅜㅁㅏ")
    print("Assemble:", assembled)
