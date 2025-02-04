export function base64Decode(str) {
  if (typeof window === "undefined") {
    return Buffer.from(str, "base64");
  } else {
    // Browser
    const binaryString = atob(str);
    const binaryData = new Uint8Array(binaryString.length);

    for (let i = 0; i < binaryString.length; i++) {
      binaryData[i] = binaryString.charCodeAt(i);
    }
    return binaryData;
  }
}
