// ai: MJPEG frames to grey through TurboJPEG, loaded at run time (2026-10-25): libturbojpeg.so.0 on Linux (its v2
// ai: calls, declared in jpeg.cpp: libjpeg-turbo 2 and 3 both export them), turbojpeg.dll or libturbojpeg.dll on
// ai: Windows, libturbojpeg.0.dylib from Homebrew's jpeg-turbo (/opt/homebrew, /usr/local) or the loader's path on
// ai: macOS, where one is installed (Media Foundation and AVFoundation decode MJPEG themselves, so the receiver needs
// ai: it on Linux alone). LIZ_TURBOJPEG, where set, is the one library tried (a path). The library is loaded once a
// ai: process for each LIZ_TURBOJPEG value load finds, and kept. TJPF_GRAY takes the luma component and skips the
// ai: chroma's IDCT; flags 0, the accurate integer IDCT. A UVC camera's frames often carry no Huffman tables (AVI1's
// ai: MJPEG); libjpeg-turbo then takes the standard ones (shown with a 4:2:2 frame whose DHT segments were cut,
// ai: 2026-10-25: decoded whole, 51 dB from its source).
// ai: The encoder is the simulated cameras' (sim/): a grey picture as a webcam sends it, 4:2:2, its tables left out.
// ai: One Jpeg a thread and a stream: TurboJPEG's handles are not shared, and libjpeg keeps a frame's Huffman tables
// ai: for the next that has none (the standard ones fill only the slots no frame defined), as a camera's own frames
// ai: are all alike.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lizard::cam {

class Jpeg {
 public:
  // ai: Null with why where the library cannot be loaded (each name tried and what the system said) or lacks a call.
  static std::unique_ptr<Jpeg> load(std::string& why);
  ~Jpeg();
  Jpeg(const Jpeg&) = delete;
  Jpeg& operator=(const Jpeg&) = delete;
  // ai: One frame's luma into out (rows of its own width, resized as needed), its size into w and h; false with why for
  // ai: a frame that does not decode: not a JPEG, a header (SOI to the first SOS) cut short or with no frame header in
  // ai: it, judged from the frame's own bytes and never from what the handle read before (TurboJPEG 3 reports the last
  // ai: header's size for a frame cut before its SOF), a fatal error, a size past 16384 a side or 2^26 pixels. A frame
  // ai: whose scan libjpeg-turbo decodes with a warning is taken, every row written, its warning in warning(): corrupt
  // ai: data, or bytes missing at its end (a frame cut short in transfer: the rows past the cut filled from the last
  // ai: data decoded, where no block reads).
  bool grey(const uint8_t* data, size_t n, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h, std::string* why);
  // ai: The warning the last grey() decoded with, "" where it had none.
  const std::string& warning() const { return warning_; }
  // ai: A grey picture (rows `stride` apart) as YCbCr 4:2:2 at quality q (1 to 100), its DHT segments cut where
  // ai: stripTables (the scan coded with the standard tables, which TurboJPEG writes unless TJ_OPTIMIZE,
  // ai: TJ_PROGRESSIVE or TJ_ARITHMETIC in the environment asks otherwise); false with why where TurboJPEG refuses.
  bool encode(const uint8_t* grey, uint32_t w, uint32_t h, uint32_t stride, int q, bool stripTables,
              std::vector<uint8_t>& out, std::string* why);

  struct Api;

 private:
  Jpeg() = default;
  const Api* api = nullptr;
  void* dec = nullptr;
  void* enc = nullptr;        // ai: made at the first encode
  std::vector<uint8_t> rgb;   // ai: encode's input, the grey picture as RGB (TurboJPEG compresses YCbCr from RGB)
  std::string warning_;
};

}  // namespace lizard::cam
