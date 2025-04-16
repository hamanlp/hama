import wasmer
import enum
import struct
from typing import List, Tuple, Dict, Optional, TypedDict, cast

# Define offsets (same as TypeScript)
DISASSEMBLE_INPUT_OFFSET: int = 0
DISASSEMBLE_INPUT_BYTE_COUNT_OFFSET: int = 4
DISASSEMBLE_IS_HANGULS_OFFSET: int = 8
DISASSEMBLE_JAMOS_OFFSET: int = 12
DISASSEMBLE_JAMOS_COUNT_OFFSET: int = 16
DISASSEMBLE_JAMOS_BYTE_COUNT_OFFSET: int = 20
DISASSEMBLE_SYLLABLE_POSITIONS_OFFSET: int = 24

ASSEMBLE_INPUT_OFFSET: int = 0  # Note: TS had 'OFFET', assuming typo for OFFSET
ASSEMBLE_INPUT_BYTE_COUNT_OFFSET: int = 4
ASSEMBLE_CHARACTERS_OFFSET: int = 8
ASSEMBLE_CHARACTERS_COUNT_OFFSET: int = 12 # This offset was unused in TS but might be relevant
ASSEMBLE_CHARACTERS_BYTE_COUNT_OFFSET: int = 16

# Define Enum (Python equivalent)
class SyllablePosition(enum.Enum):
    CODA = 0
    NUCLEUS = 1
    ONSET = 2
    NOT_APPLICABLE = 3

# Define Result Types using TypedDict for clarity
class DisassembleResult(TypedDict):
    input: str
    is_hanguls: List[bool]
    text: str  # Represents the jamos
    syllable_positions: List[SyllablePosition]

class AssembleResult(TypedDict):
    input: str
    text: str # Represents the assembled characters

# --- Helper function for logging from WASM ---
def py_log(x: int):
    """Equivalent of jslog for WASM to call."""
    print(f"WASM log: {x}")

class JamoParser:
        self.wasm_file_path = Path(__file__).parent / "hama-g2p.wasm"
        self.store: Optional[wasmer.Store] = None
        self.instance: Optional[wasmer.Instance] = None
        self.memory: Optional[wasmer.Memory] = None
        # Exported functions
        self._disassemble = None
        self._cleanup_disassemble = None
        self._assemble = None
        self._cleanup_assemble = None
        self._alloc_uint8 = None
        self.loaded: bool = False

    def load(self) -> None:
        """Loads and instantiates the WASM module."""
        if self.loaded:
            print("WASM module already loaded.")
            return

        try:
            print(f"Loading WASM module from: {self.wasm_file_path}")
            self.store = wasmer.Store()
            module = wasmer.Module(self.store, open(self.wasm_file_path, 'rb').read())

            # --- Define imports ---
            # Corresponds to `importObject` in TypeScript
            import_object = {
                "env": {
                    # Make the Python function available to WASM
                    "jslog": wasmer.Function(self.store, py_log)
                    # Add other imported functions if your WASM requires them
                }
            }

            self.instance = wasmer.Instance(module, import_object)
            self.memory = self.instance.exports.memory

            # --- Get exported functions ---
            # Use getattr for slightly safer access
            self._disassemble = getattr(self.instance.exports, 'disassemble', None)
            self._assemble = getattr(self.instance.exports, 'assemble', None)
            self._cleanup_disassemble = getattr(self.instance.exports, 'cleanup_disassemble', None)
            self._cleanup_assemble = getattr(self.instance.exports, 'cleanup_assemble', None)
            self._alloc_uint8 = getattr(self.instance.exports, 'allocUint8', None) # Match TS name

            # --- Basic validation ---
            if not all([self.memory, self._disassemble, self._assemble,
                        self._cleanup_disassemble, self._cleanup_assemble,
                        self._alloc_uint8]):
                raise RuntimeError("One or more required exports not found in WASM module.")

            self.loaded = True
            print("WASM module loaded successfully.")

        except Exception as e:
            print(f"Error loading WASM module: {e}")
            # Reset state
            self.store = None
            self.instance = None
            self.memory = None
            self._disassemble = None
            self._cleanup_disassemble = None
            self._assemble = None
            self._cleanup_assemble = None
            self._alloc_uint8 = None
            self.loaded = False
            raise # Re-raise the exception

    def _check_loaded(self) -> None:
        """Raises RuntimeError if WASM module is not loaded."""
        if not self.loaded or self.instance is None or self.memory is None:
            raise RuntimeError("WASM module is not loaded. Call load() first.")
        if self._alloc_uint8 is None:
             raise RuntimeError("WASM function '_alloc_uint8' is not available.")

    def _encode_string(self, text: str) -> Tuple[int, int]:
        """Encodes string to UTF-8, allocates WASM memory, copies data."""
        self._check_loaded()
        encoded_bytes = text.encode('utf-8')
        byte_length = len(encoded_bytes)

        # Allocate memory in WASM
        pointer = self._alloc_uint8(byte_length)
        if not isinstance(pointer, int) or pointer == 0:
             raise MemoryError("Failed to allocate memory in WASM.")

        # Create a memory view and copy data
        memory_view = self.memory.uint8_view(offset=pointer) # type: ignore # Memory checked in _check_loaded
        if len(memory_view) < byte_length:
            raise MemoryError(f"Allocated WASM memory ({len(memory_view)} bytes) is smaller than required ({byte_length} bytes).")

        memory_view[0:byte_length] = encoded_bytes # Copy bytes

        return pointer, byte_length

    def _decode_string(self, pointer: int, length: int) -> str:
        """Reads UTF-8 string from WASM memory."""
        self._check_loaded()
        memory_view = self.memory.uint8_view(offset=pointer) # type: ignore
        if length == 0:
            return ""
        if len(memory_view) < length:
             raise MemoryError(f"Attempting to read {length} bytes from WASM memory, but view only has {len(memory_view)} bytes starting at offset {pointer}.")

        # Get the relevant bytes and decode
        byte_slice = bytes(memory_view[0:length])
        return byte_slice.decode('utf-8')

    def _read_uint32(self, address: int) -> int:
        """Reads a little-endian unsigned 32-bit integer from WASM memory."""
        self._check_loaded()
        # Use struct.unpack_from for safe reading at specific offsets
        # '<I' means little-endian unsigned int (4 bytes)
        buffer = self.memory.buffer # type: ignore
        try:
            return struct.unpack_from("<I", buffer, address)[0]
        except struct.error as e:
            raise MemoryError(f"Failed to read uint32 at address {address}: {e}. Buffer size: {len(buffer)}")
        except IndexError as e:
             raise MemoryError(f"IndexError reading uint32 at address {address}: {e}. Buffer size: {len(buffer)}")


    def _read_uint8_array(self, address: int, count: int) -> bytes:
        """Reads a sequence of bytes from WASM memory."""
        self._check_loaded()
        memory_view = self.memory.uint8_view(offset=address) # type: ignore
        if count == 0:
            return b""
        if len(memory_view) < count:
            raise MemoryError(f"Attempting to read {count} bytes from WASM memory, but view only has {len(memory_view)} bytes starting at offset {address}.")
        return bytes(memory_view[0:count])

    def disassemble(self, text: str) -> DisassembleResult:
        """Disassembles Hangul text into Jamos using WASM."""
        self._check_loaded()
        if self._disassemble is None or self._cleanup_disassemble is None:
             raise RuntimeError("WASM function 'disassemble' or 'cleanup_disassemble' is not available.")

        # 1. Encode input string and copy to WASM memory
        input_ptr, input_len = self._encode_string(text)
        result_ptr = 0 # Initialize in case of early exit in finally

        try:
            # 2. Call the WASM disassemble function
            # Assuming the third argument corresponds to the 'true' in TS
            result_ptr = self._disassemble(input_ptr, input_len, True)
            if not isinstance(result_ptr, int) or result_ptr == 0:
                raise RuntimeError(f"WASM 'disassemble' function returned an invalid pointer: {result_ptr}")

            # 3. Read the result structure from WASM memory
            # Use helper to read uint32 values safely
            input_address = self._read_uint32(result_ptr + DISASSEMBLE_INPUT_OFFSET)
            input_byte_count = self._read_uint32(result_ptr + DISASSEMBLE_INPUT_BYTE_COUNT_OFFSET)
            is_hanguls_address = self._read_uint32(result_ptr + DISASSEMBLE_IS_HANGULS_OFFSET)
            jamos_address = self._read_uint32(result_ptr + DISASSEMBLE_JAMOS_OFFSET)
            jamos_count = self._read_uint32(result_ptr + DISASSEMBLE_JAMOS_COUNT_OFFSET)
            jamos_byte_count = self._read_uint32(result_ptr + DISASSEMBLE_JAMOS_BYTE_COUNT_OFFSET)
            syllable_positions_address = self._read_uint32(result_ptr + DISASSEMBLE_SYLLABLE_POSITIONS_OFFSET)

            # 4. Decode/process the data
            # Decode original input string
            original_input = self._decode_string(input_address, input_byte_count)

            # Get is_hanguls (array of booleans)
            is_hanguls_raw = self._read_uint8_array(is_hanguls_address, jamos_count)
            is_hanguls = [bool(value) for value in is_hanguls_raw]

            # Decode jamos string
            jamos = self._decode_string(jamos_address, jamos_byte_count)

            # Get syllable positions (array of enums)
            syllable_positions_raw = self._read_uint8_array(syllable_positions_address, jamos_count)
            syllable_positions: List[SyllablePosition] = []
            for value in syllable_positions_raw:
                try:
                    syllable_positions.append(SyllablePosition(value))
                except ValueError:
                    # Keep original error handling logic
                    raise ValueError(f"Invalid syllable position value: {value}")

            # 5. Construct result dictionary
            result: DisassembleResult = {
                "input": original_input,
                "text": jamos,
                "is_hanguls": is_hanguls,
                "syllable_positions": syllable_positions,
            }
            return result

        finally:
            # 6. Clean up WASM memory allocated by disassemble
            if result_ptr != 0 and self._cleanup_disassemble:
                self._cleanup_disassemble(result_ptr)
            # Note: We don't explicitly free the memory allocated by _encode_string
            # here. It's assumed the WASM 'disassemble' function might reuse or
            # take ownership, or that its own cleanup handles it, or that memory
            # is managed more broadly (e.g., linear memory grows). If explicit
            # freeing is needed, the WASM module should export a 'free' function.


    def assemble(self, text: str) -> AssembleResult:
        """Assembles Jamos text into Hangul characters using WASM."""
        self._check_loaded()
        if self._assemble is None or self._cleanup_assemble is None:
             raise RuntimeError("WASM function 'assemble' or 'cleanup_assemble' is not available.")

        # 1. Encode input jamo string and copy to WASM memory
        input_ptr, input_len = self._encode_string(text)
        result_ptr = 0 # Initialize

        try:
            # 2. Call the WASM assemble function
            result_ptr = self._assemble(input_ptr, input_len)
            if not isinstance(result_ptr, int) or result_ptr == 0:
                 raise RuntimeError(f"WASM 'assemble' function returned an invalid pointer: {result_ptr}")

            # 3. Read the result structure from WASM memory
            input_address = self._read_uint32(result_ptr + ASSEMBLE_INPUT_OFFSET)
            input_byte_count = self._read_uint32(result_ptr + ASSEMBLE_INPUT_BYTE_COUNT_OFFSET)
            characters_address = self._read_uint32(result_ptr + ASSEMBLE_CHARACTERS_OFFSET)
            characters_byte_count = self._read_uint32(result_ptr + ASSEMBLE_CHARACTERS_BYTE_COUNT_OFFSET)
            # characters_count = self._read_uint32(result_ptr + ASSEMBLE_CHARACTERS_COUNT_OFFSET) # Read if needed

            # 4. Decode the data
            original_input = self._decode_string(input_address, input_byte_count)
            assembled_text = self._decode_string(characters_address, characters_byte_count)

            # 5. Construct result dictionary
            result: AssembleResult = {
                "input": original_input,
                "text": assembled_text,
            }
            return result

        finally:
            # 6. Clean up WASM memory allocated by assemble
            if result_ptr != 0 and self._cleanup_assemble:
                self._cleanup_assemble(result_ptr)
            # See cleanup note in disassemble


# --- Example Usage ---
if __name__ == "__main__":
    try:
        parser = JamoParser()
        parser.load()

        # --- Test Disassemble ---
        hangul_text = "안녕하세요"
        print(f"Disassembling: '{hangul_text}'")
        disassembled_result = parser.disassemble(hangul_text)
        print("Result:")
        # Pretty print the result
        print(f"  Input: {disassembled_result['input']}")
        print(f"  Jamos: {disassembled_result['text']}")
        print(f"  Is Hanguls: {disassembled_result['is_hanguls']}")
        syllable_names = [pos.name for pos in disassembled_result['syllable_positions']]
        print(f"  Syllable Positions: {syllable_names}")
        print("-" * 20)

        jamo_text = disassembled_result['text'] # Use output from disassemble

        # --- Test Assemble ---
        print(f"Assembling: '{jamo_text}'")
        assembled_result = parser.assemble(jamo_text)
        print("Result:")
        print(f"  Input: {assembled_result['input']}")
        print(f"  Assembled Text: {assembled_result['text']}")
        print("-" * 20)

        # --- Test Non-Hangul ---
        mixed_text = "Hello 안녕"
        print(f"Disassembling: '{mixed_text}'")
        disassembled_mixed = parser.disassemble(mixed_text)
        print("Result:")
        print(f"  Input: {disassembled_mixed['input']}")
        print(f"  Jamos/Chars: {disassembled_mixed['text']}")
        print(f"  Is Hanguls: {disassembled_mixed['is_hanguls']}")
        syllable_names_mixed = [pos.name for pos in disassembled_mixed['syllable_positions']]
        print(f"  Syllable Positions: {syllable_names_mixed}")
        print("-" * 20)

        print(f"Assembling: '{disassembled_mixed['text']}'")
        assembled_mixed = parser.assemble(disassembled_mixed['text'])
        print("Result:")
        print(f"  Input: {assembled_mixed['input']}")
        print(f"  Assembled Text: {assembled_mixed['text']}")
        print("-" * 20)


    except FileNotFoundError:
        print("\nError: hama.wasm not found. Please place the WASM file in the same directory or provide the correct path.")
    except (RuntimeError, MemoryError, ValueError, ImportError) as e:
        print(f"\nAn error occurred: {e}")
        # If wasmer isn't installed
        if isinstance(e, ImportError) and 'wasmer' in str(e):
            print("Hint: Make sure you have installed the wasmer library: pip install wasmer wasmer_compiler_cranelift")
