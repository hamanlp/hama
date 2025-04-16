import wasmer
from pathlib import Path
import struct
from typing import Tuple, Optional, cast
import logging # Use logging instead of print for errors/info
import os # For context manager support

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define offsets (same as TypeScript)
G2P_INPUT_OFFSET: int = 0
G2P_INPUT_BYTE_COUNT_OFFSET: int = 4
G2P_IPA_OFFSET: int = 8
G2P_IPA_BYTE_COUNT_OFFSET: int = 12

# --- Helper function for logging from WASM ---
def py_log(x: int):
    """Equivalent of jslog for WASM to call."""
    print(f"WASM log: {x}") # Keep simple print for direct WASM output

class Phonemizer:
    def __init__(self):
        """
        Initializes the Phonemizer wrapper.
        """
        wasm_file_path = Path(__file__).parent / "hama-g2p.wasm"
        if not os.path.exists(wasm_file_path):
             raise FileNotFoundError(f"WASM file not found at: {wasm_file_path}")
        self.wasm_file_path = wasm_file_path
        self.store: Optional[wasmer.Store] = None
        self.instance: Optional[wasmer.Instance] = None
        self.memory: Optional[wasmer.Memory] = None

        # Exported functions
        self._init_phonemizer = None
        self._to_ipa = None
        self._alloc_uint8 = None
        self._deinit_result = None
        self._deinit_phonemizer = None

        # Internal state pointer from WASM
        self.phonemizer_ptr: Optional[int] = None
        self.loaded: bool = False

    def load(self) -> None:
        """
        Loads the WASM module, instantiates it, and initializes the phonemizer state.
        Raises RuntimeError if loading or initialization fails.
        """
        if self.loaded:
            logging.info("Phonemizer WASM module already loaded.")
            return

        try:
            logging.info(f"Loading Phonemizer WASM module from: {self.wasm_file_path}")
            self.store = wasmer.Store()
            module = wasmer.Module(self.store, open(self.wasm_file_path, 'rb').read())

            # --- Define imports ---
            import_object = {
                "env": {
                    "jslog": wasmer.Function(self.store, py_log)
                }
            }

            self.instance = wasmer.Instance(module, import_object)
            self.memory = self.instance.exports.memory

            # --- Get exported functions ---
            exports = self.instance.exports
            self._init_phonemizer = getattr(exports, 'init_phonemizer', None)
            self._to_ipa = getattr(exports, 'to_ipa', None)
            self._alloc_uint8 = getattr(exports, 'allocUint8', None) # Match TS name
            self._deinit_result = getattr(exports, 'deinit_result', None)
            self._deinit_phonemizer = getattr(exports, 'deinit_phonemizer', None)

            # --- Basic validation ---
            if not all([self.memory, self._init_phonemizer, self._to_ipa,
                        self._alloc_uint8, self._deinit_result, self._deinit_phonemizer]):
                missing = [name for name, func in [
                    ('memory', self.memory), ('init_phonemizer', self._init_phonemizer),
                    ('to_ipa', self._to_ipa), ('allocUint8', self._alloc_uint8),
                    ('deinit_result', self._deinit_result),
                    ('deinit_phonemizer', self._deinit_phonemizer)] if func is None]
                raise RuntimeError(f"One or more required exports not found in WASM module: {', '.join(missing)}")

            # --- Initialize internal WASM phonemizer state ---
            logging.info("Initializing WASM phonemizer state...")
            self.phonemizer_ptr = self._init_phonemizer()
            if not isinstance(self.phonemizer_ptr, int) or self.phonemizer_ptr == 0:
                # Clean up instance if init fails right away
                self.instance = None
                self.memory = None
                raise RuntimeError(f"WASM 'init_phonemizer' failed or returned an invalid pointer: {self.phonemizer_ptr}")

            self.loaded = True
            logging.info("Phonemizer WASM module loaded and initialized successfully.")

        except Exception as e:
            logging.error(f"Error loading or initializing Phonemizer WASM module: {e}", exc_info=True)
            self._cleanup_resources() # Ensure cleanup on error
            raise # Re-raise the exception

    def _check_loaded_and_initialized(self) -> None:
        """Raises RuntimeError if WASM module is not loaded and initialized."""
        if not self.loaded or self.instance is None or self.memory is None:
            raise RuntimeError("Phonemizer WASM module is not loaded. Call load() first or use a 'with' statement.")
        if self.phonemizer_ptr is None or self.phonemizer_ptr == 0:
            raise RuntimeError("WASM phonemizer state is not initialized correctly.")
        if self._alloc_uint8 is None:
             raise RuntimeError("WASM function '_alloc_uint8' is not available.")


    def _encode_string(self, text: str) -> Tuple[int, int]:
        """Encodes string to UTF-8, allocates WASM memory, copies data."""
        self._check_loaded_and_initialized()
        encoded_bytes = text.encode('utf-8')
        byte_length = len(encoded_bytes)
        if byte_length == 0:
            # Handle empty string - allocate 1 byte for null terminator?
            # Or handle in WASM? Let's allocate 0, WASM should handle it.
            # pointer = self._alloc_uint8(1) ... memory_view[0] = 0
             # Allocate 0 bytes, return pointer 0? Let's try allocating 0.
             # WASM function needs to be robust to 0 length input.
             # Safest is maybe allocate 1 byte if WASM expects null termination?
             # For now, assume WASM handles len 0. If issues arise, revisit.
            pointer = self._alloc_uint8(0)
            # We might actually need a valid pointer even for zero length. Let's allocate 1 byte.
            if byte_length == 0:
                 pointer = self._alloc_uint8(1)
                 memory_view = self.memory.uint8_view(offset=pointer)
                 memory_view[0] = 0 # Null terminate explicitly?
                 return pointer, 0 # Return length 0 still

        # Allocate memory in WASM
        pointer = self._alloc_uint8(byte_length)
        if not isinstance(pointer, int) or pointer == 0:
             raise MemoryError(f"Failed to allocate {byte_length} bytes in WASM.")

        # Create a memory view and copy data
        memory_view = self.memory.uint8_view(offset=pointer) # type: ignore
        if len(memory_view) < byte_length:
            # This should ideally not happen if allocUint8 worked correctly
            self._free_encoded_string(pointer) # Attempt to free allocated memory
            raise MemoryError(f"Allocated WASM memory ({len(memory_view)} bytes at {pointer}) is smaller than required ({byte_length} bytes).")

        try:
            memory_view[0:byte_length] = encoded_bytes # Copy bytes
        except Exception as e:
            self._free_encoded_string(pointer) # Attempt to free if copy fails
            raise MemoryError(f"Failed to copy data to WASM memory at {pointer}: {e}")

        return pointer, byte_length

    def _free_encoded_string(self, pointer: int):
         """Placeholder: Free memory allocated by _encode_string if WASM provides a 'free' function."""
         # If your WASM exports a 'free' or similar function:
         # try:
         #     _free_func = getattr(self.instance.exports, 'free', None)
         #     if _free_func and pointer != 0:
         #         _free_func(pointer)
         # except Exception as e:
         #     logging.warning(f"Failed to free WASM memory at {pointer}: {e}")
         pass # Assuming allocUint8 memory is managed by the caller functions (to_ipa/deinit_result)


    def _decode_string(self, pointer: int, length: int) -> str:
        """Reads UTF-8 string from WASM memory."""
        self._check_loaded_and_initialized()
        if length == 0:
            return ""
        if pointer == 0:
            logging.warning("Attempting to decode string from null pointer (address 0). Returning empty string.")
            return ""

        memory_view = self.memory.uint8_view(offset=pointer) # type: ignore
        if len(memory_view) < length:
             raise MemoryError(f"Attempting to read {length} bytes from WASM memory, but view only has {len(memory_view)} bytes starting at offset {pointer}.")

        # Get the relevant bytes and decode
        try:
            byte_slice = bytes(memory_view[0:length])
            return byte_slice.decode('utf-8')
        except UnicodeDecodeError as e:
            logging.error(f"Failed to decode UTF-8 string from WASM memory at {pointer} (length {length}): {e}")
            # Optionally return raw bytes or raise error
            return f"<{len(byte_slice)} bytes decoding error>"
        except Exception as e:
             raise MemoryError(f"Error reading string from WASM memory at {pointer}: {e}")


    def _read_uint32(self, address: int) -> int:
        """Reads a little-endian unsigned 32-bit integer from WASM memory."""
        self._check_loaded_and_initialized()
        if address < 0:
             raise ValueError(f"Attempting to read uint32 from negative address: {address}")

        buffer = self.memory.buffer # type: ignore
        # Check bounds BEFORE reading
        current_memory_size = self.memory.data_size
        if address + 4 > current_memory_size:
             raise MemoryError(f"Attempting to read uint32 at address {address}, but it would exceed buffer size {current_memory_size}.")

        try:
            # '<I' means little-endian unsigned int (4 bytes)
            return struct.unpack_from("<I", buffer, address)[0]
        except struct.error as e:
            # This might happen if memory is invalid or address is wrong despite bounds check
            raise MemoryError(f"Struct unpacking error reading uint32 at address {address}: {e}. Buffer size: {len(buffer)}")
        except IndexError:
            # unpack_from might raise IndexError theoretically, though bounds check should prevent it
             raise MemoryError(f"IndexError reading uint32 at address {address}. Buffer size: {len(buffer)}")


    def to_ipa(self, text: str) -> str:
        """
        Converts Korean text to its IPA representation using the WASM module.

        Args:
            text: The Korean text to convert.

        Returns:
            The IPA string.

        Raises:
            RuntimeError: If the WASM module is not loaded/initialized or if conversion fails.
            MemoryError: If memory allocation or access fails.
        """
        self._check_loaded_and_initialized()
        if self._to_ipa is None or self._deinit_result is None:
             raise RuntimeError("WASM function '_to_ipa' or '_deinit_result' is not available.")

        input_ptr, input_len = 0, 0
        result_ptr: Optional[int] = 0 # Use Optional[int] for clarity

        try:
            # 1. Encode input string and copy to WASM memory
            input_ptr, input_len = self._encode_string(text)

            # 2. Call the WASM to_ipa function
            # Pass the phonemizer state pointer obtained during load/init
            result_ptr = self._to_ipa(self.phonemizer_ptr, input_ptr, input_len)
            if not isinstance(result_ptr, int) or result_ptr == 0:
                raise RuntimeError(f"WASM 'to_ipa' function failed or returned an invalid result pointer: {result_ptr}")

            # 3. Read the result structure from WASM memory
            # We only need the IPA string part based on the TS code
            ipa_address = self._read_uint32(result_ptr + G2P_IPA_OFFSET)
            ipa_byte_count = self._read_uint32(result_ptr + G2P_IPA_BYTE_COUNT_OFFSET)

            # --- Optional: Read back original input for verification (like TS) ---
            # input_address = self._read_uint32(result_ptr + G2P_INPUT_OFFSET)
            # input_byte_count = self._read_uint32(result_ptr + G2P_INPUT_BYTE_COUNT_OFFSET)
            # original_input_from_wasm = self._decode_string(input_address, input_byte_count)
            # if original_input_from_wasm != text:
            #     logging.warning(f"Input mismatch: provided='{text}', read_back='{original_input_from_wasm}'")
            # --------------------------------------------------------------------

            # 4. Decode the IPA string
            ipa_result = self._decode_string(ipa_address, ipa_byte_count)

            return ipa_result

        except Exception as e:
            logging.error(f"Error during 'to_ipa' conversion for text '{text[:50]}...': {e}", exc_info=True)
            raise # Re-raise after logging

        finally:
            # 5. Clean up WASM memory allocated by to_ipa (the result structure)
            # This happens regardless of whether processing the result succeeded or failed
            if result_ptr is not None and result_ptr != 0 and self._deinit_result:
                 try:
                     # Pass the phonemizer state pointer AND the result pointer to deinit
                     self._deinit_result(self.phonemizer_ptr, result_ptr)
                 except Exception as e:
                     # Log if cleanup fails, but don't prevent potential error propagation from 'try' block
                     logging.error(f"WASM 'deinit_result' failed for pointer {result_ptr}: {e}", exc_info=True)

            # 6. Clean up WASM memory allocated for the input string
            # This assumes the WASM `to_ipa` doesn't take ownership or free it itself.
            # If WASM doesn't provide 'free', this memory might leak per call, unless
            # `deinit_result` or `deinit_phonemizer` handles it implicitly.
            # Adding a placeholder call. Remove if WASM manages this memory.
            if input_ptr != 0:
                self._free_encoded_string(input_ptr)


    def close(self) -> None:
        """
        Deinitializes the WASM phonemizer state and releases resources.
        Safe to call multiple times or on an unloaded instance.
        """
        if self.phonemizer_ptr is not None and self.phonemizer_ptr != 0 and self._deinit_phonemizer:
            logging.info(f"Deinitializing WASM phonemizer state (pointer: {self.phonemizer_ptr})...")
            try:
                self._deinit_phonemizer(self.phonemizer_ptr)
            except Exception as e:
                logging.error(f"Error during WASM 'deinit_phonemizer': {e}", exc_info=True)
            finally:
                 # Ensure pointer is cleared even if deinit fails
                self.phonemizer_ptr = None
        elif self.loaded:
             logging.warning("Attempting to close Phonemizer, but phonemizer pointer is invalid or deinit function missing.")

        # Clean up other resources
        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Resets internal state variables."""
        logging.debug("Cleaning up Phonemizer Python-side resources.")
        self.instance = None
        self.memory = None
        self.store = None # Store might hold onto resources
        # Clear function references
        self._init_phonemizer = None
        self._to_ipa = None
        self._alloc_uint8 = None
        self._deinit_result = None
        self._deinit_phonemizer = None
        # Ensure phonemizer_ptr is cleared if not already
        self.phonemizer_ptr = None
        self.loaded = False

    # --- Context Manager Protocol ---
    def __enter__(self) -> 'Phonemizer':
        """Load WASM and initialize when entering 'with' block."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup WASM state when exiting 'with' block."""
        self.close()

# --- Example Usage ---
if __name__ == "__main__":

    korean_text = "안녕하세요 세계"
    try:
        # Using the context manager ensures cleanup
        print(f"\nAttempting to phonemize: '{korean_text}'")
        with Phonemizer() as phonemizer:
            ipa_result = phonemizer.to_ipa(korean_text)
            print(f"\nInput:  '{korean_text}'")
            print(f"Output (IPA): '{ipa_result}'")

            # Test another conversion
            korean_text_2 = "감사합니다"
            print(f"\nInput:  '{korean_text_2}'")
            ipa_result_2 = phonemizer.to_ipa(korean_text_2)
            print(f"Output (IPA): '{ipa_result_2}'")

        print("\nPhonemizer resources released via 'with' statement.")

        # Example of manual load/close (less recommended)
        print("\n--- Manual Load/Close Example ---")
        manual_phonemizer = Phonemizer()
        try:
                manual_phonemizer.load()
                ipa_manual = manual_phonemizer.to_ipa("테스트")
                print(f"Input: '테스트'")
                print(f"Output (IPA): '{ipa_manual}'")
        finally:
                manual_phonemizer.close()
                print("Manual Phonemizer resources released.")


    except FileNotFoundError as e:
            print(f"\nError: {e}") # Already handled above, but good practice
    except (RuntimeError, MemoryError, ValueError, ImportError) as e:
        print(f"\nAn error occurred during phonemization: {e}")
        # If wasmer isn't installed
        if isinstance(e, ImportError) and 'wasmer' in str(e):
            print("Hint: Make sure you have installed the wasmer library: pip install wasmer wasmer_compiler_cranelift")
    except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
