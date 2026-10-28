// ai: .lizard (2026-10-28): a replay as one file, the only form a replay takes (Save replays writes it, the phone's
// ai: Download is it, the PC's Upload camera plays it, the lab's tools read it; research/SPEC.md 15). A ZIP archive of
// ai: stored entries (method 0, no ZIP64, so under 4 GB and 65,535 entries), its comment "LIZARD replay 1": each frame
// ai: as NNNN.zst, in the run's order; then stats.jsonl where the run has rows; then meta.json, which ends it (frames,
// ai: sizes, ms, axes, lens, layout, ...: replay.h finish). A reader finds the entries by the central directory. A frame's
// ai: entry is its w x h luma (meta.json's sizes[i], else w and h) as the residuals of JPEG-LS's median predictor
// ai: (LOCO-I's MED: of a, b, c, the neighbours left, above and above-left, min(a, b) where c >= max(a, b), max(a, b)
// ai: where c <= min(a, b), else a + b - c; a neighbour outside the frame 0), each the pixel less its prediction mod 256,
// ai: the w h of them in rows as one zstd frame at level 1. Lossless; on the S26's 2:1 frames 74.4% of the raw bytes
// ai: (gzip -1 88.4%), about 1.4 ms a 960 x 960 frame to pack on its big core, 2.0 to unpack.
#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace lizard::lizfile {

// ai: a frame's luma (rows `stride` apart) as its NNNN.zst entry, and back: false with why where the bytes are not one
// ai: zstd frame of w h residuals
std::vector<uint8_t> packFrame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride);
bool unpackFrame(const uint8_t* z, size_t n, uint32_t w, uint32_t h, uint8_t* out, std::string& why);

// ai: A .lizard written as it goes: the frames in order, then finish's stats.jsonl and meta.json. Written through
// ai: path.part, renamed over path at finish; one destroyed unfinished removes its .part. One thread at a time.
class Writer {
 public:
  static std::unique_ptr<Writer> create(const std::string& path, std::string& why);
  ~Writer();
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  // ai: false with why (a write the file system refused, ZIP's 4 GB): the archive keeps the frames before it, so a
  // ai: finish after it still makes a whole file of them
  bool frame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, std::string& why);
  size_t frames() const { return frames_; }
  // ai: the file's bytes, -1 with why; stats "" none
  int64_t finish(const std::string& meta, const std::string& stats, std::string& why);

 private:
  Writer() = default;
  std::string path_;
  std::FILE* f_ = nullptr;
  void* cctx_ = nullptr;
  void* zip_ = nullptr;
  size_t frames_ = 0;
  bool done_ = false;
};

// ai: A .lizard as a folder of its parts (meta.json, stats.jsonl, NNNN.gray: the frames as raw luma), to look at them
// ai: with other tools, and such a folder as a .lizard. False / -1 with why; unpack refuses a folder that holds anything.
bool unpack(const std::string& path, const std::string& dir, std::string& why);
int64_t pack(const std::string& dir, const std::string& path, std::string& why);

// ai: A .lizard to read a frame at a time. One thread at a time.
struct FrameInfo {
  uint32_t w = 0, h = 0;
  double ms = 0;      // ai: its capture's time on the camera's clock; NaN where meta.json has none
  float ax = 0, ay = 0;   // ai: where the camera's optical axis fell in it; NaN its centre (meta.json's axes)
};
class Reader {
 public:
  static std::unique_ptr<Reader> open(const std::string& path, std::string& why);
  ~Reader();
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  const std::string& meta() const { return meta_; }   // ai: meta.json as the run wrote it
  const std::string& stats() const { return stats_; }   // ai: stats.jsonl, "" where the run has none
  const std::string& layout() const { return layout_; }   // ai: meta.json's layout, "1:1" or "2:1" ("1:1" where none)
  const std::vector<FrameInfo>& frames() const { return frames_; }
  // ai: frame i's luma, w h bytes in rows of w; false with why (a CRC that does not hold, a short file)
  bool frame(size_t i, std::vector<uint8_t>& luma, std::string& why);

 private:
  Reader() = default;
  struct Entry { uint64_t at = 0, size = 0; uint32_t crc = 0; };
  std::string path_, meta_, stats_, layout_ = "1:1";
  std::FILE* f_ = nullptr;
  void* dctx_ = nullptr;         // ai: zstd's decompression context, made at the first frame
  std::vector<Entry> entries_;   // ai: a .lizard's frames' entries, in order
  std::vector<FrameInfo> frames_;
  std::vector<uint8_t> buf_;
  bool parseMeta(std::string& why);
  bool readEntry(const Entry& e, std::vector<uint8_t>& out, std::string& why);
};

}  // namespace lizard::lizfile
