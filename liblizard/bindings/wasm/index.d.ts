// ai: Types for index.mjs, liblizard's WebAssembly binding (the C API in ../../include/lizard.h says what each does).

export declare function init(moduleOptions?: object): Promise<unknown>;
export declare class LizardError extends Error {
  readonly name: "LizardError";
  /** One of lizard.h's LIZ_E_* codes (negative). */
  readonly code: number;
}

export declare const BLOCK: 473, ID_BYTES: 4, PAYLOAD: 469, MAX_BLOCKS: 128, RINGS: 4, RING_DEFAULT: -1, GAP_MODULES: 12;
export type PixelFormat = "grey" | "rgbx" | "rgba" | "bgra";

export declare function version(): string;
export declare function abi(): number;
/** Whether the codec's vector paths (WebAssembly SIMD) are in this build. */
export declare function simd(): boolean;

export interface Format {
  /** The format's size, 1 to 128 (LIZARD-8 to LIZARD-1024); a symbol carries the rate profile's blocks
   *  (frameBlocks over codes). */
  blocks: number;
  /** 0 to 3 (the 32, 64, 96 or 128 ring), -1 the default (the 128). */
  ring?: number;
  /** The display rate the symbol's word states, 1 to 255 (default 60). */
  fps?: number;
  /** 1, or 2 symbols side by side (default 1). */
  codes?: 1 | 2;
}
export interface Geometry {
  n: number; span: number; pxm: number; side: number; width: number; height: number; gap: number; frameBlocks: number;
}
export declare function ringCells(ring?: number): number;
export declare function geometry(format: Format): Geometry;
export declare function roomFor(blocks: number, ring?: number): number;
export declare function pick(w: number, h: number, options?: { codes?: 1 | 2; ring?: number; top?: number }): number;

export declare class Encoder {
  constructor(format: Format);
  readonly format: Required<Format>;
  readonly geometry: Geometry;
  /** blocks: frameBlocks x 473 bytes; picture: the frame's count (its value mod 4 the pilots). */
  paint<T extends Uint8Array | Uint8ClampedArray = Uint8ClampedArray>(blocks: Uint8Array, picture: number, out?: T, fmt?: PixelFormat): T;
  free(): void;
}

export interface Decoded {
  /** Blocks verified, and their bytes (count x 473). */
  count: number;
  blocks: Uint8Array;
  found: boolean;
  ring: number;
  n: number;
  word: boolean;
  blocksPerSymbol: number;
  fps: number;
  heldUsed: boolean;
  total: number;
  /** Corners in the image: x y x y ..., top left, top right, bottom right, bottom left. */
  quad: number[];
  pilot: { blocks: number; r: number[]; sd: number[] };
}
export interface ImageLike { data: Uint8Array | Uint8ClampedArray; width: number; height: number }
export declare class Decoder {
  constructor(nmax?: number);
  readonly maxBlocks: number;
  /** The held word: the size (Format.blocks) the last word read names (0 none). */
  held: number;
  decode(px: Uint8Array | Uint8ClampedArray | ImageLike, w?: number, h?: number,
         options?: { fmt?: PixelFormat; stride?: number; held?: { held: number } }): Decoded;
  free(): void;
}
export declare function layoutRects(w: number, h: number, layout?: 1 | 2): { x: number; y: number; w: number; h: number }[];
export declare function streamFill(id: number): Uint8Array;

export interface TxInfo { test: boolean; length: number; chunks: number; lap: number; root: string }
export declare class Tx {
  static file(bytes: Uint8Array, name?: string, type?: string): Tx;
  static test(firstId?: number): Tx;
  next(n: number): Uint8Array;
  readonly info: TxInfo;
  free(): void;
}

export interface Verdict { seen: number; bad: number; judged: number; fresh: number; test: boolean }
export interface Progress {
  header: boolean; done: boolean; length: number; bytesIn: number; fraction: number;
  chunks: number; verified: number; rejected: number; solveMs: number;
}
export declare class Rx {
  constructor(options?: { maxBytes?: number });
  frame(blocks: Uint8Array): Verdict;
  readonly progress: Progress;
  readonly chunks: Uint8Array;
  readonly name: string;
  readonly type: string;
  readonly root: string;
  readonly error: string;
  data(): Uint8Array | null;
  clear(): void;
  free(): void;
}

export declare class Sender {
  constructor(source: { bytes: Uint8Array; name?: string; type?: string } | { test: true; firstId?: number }, format: Format);
  readonly geometry: Geometry;
  readonly info: TxInfo;
  picture: number;
  frame<T extends Uint8Array | Uint8ClampedArray = Uint8ClampedArray>(fmt?: PixelFormat, out?: T): { data: T; width: number; height: number; picture: number };
  free(): void;
}
export declare class Receiver {
  constructor(options?: { layout?: 1 | 2; nmax?: number; maxBytes?: number });
  readonly decoder: Decoder;
  readonly rx: Rx;
  push(px: Uint8Array | Uint8ClampedArray | ImageLike, w?: number, h?: number, options?: { fmt?: PixelFormat; stride?: number }):
    { decoded: Decoded[]; verdict: Verdict; progress: Progress };
  readonly file: { name: string; type: string; root: string; bytes: Uint8Array } | null;
  clear(): void;
  free(): void;
}
