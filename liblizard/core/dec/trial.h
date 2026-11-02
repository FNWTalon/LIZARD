// ai: The forms' trial (2026-10-31): a GPU decoder is used only in forms that read known frames right on this device.
// ai: Frames painted by the C (src/lizard_phy.h) with blocks of known bytes, one a picture size the setup holds, the rings in
// ai: turn, under a fixed noise so the LDPC iterates, are decoded in each variant, each LDPC form and the classifier's
// ai: int8 twin; a form passes when every painted block comes back verified with its bytes, and is refused when one is
// ai: missing or verified wrong, or when the device hangs or is lost in it (wg::Lost). A refused form is left out (the
// ai: next form; the next variant where no LDPC form passes; the C where no variant does), and the first batch's timing
// ai: (front.cpp measureLdpc, measureTwins) picks among those that passed. A feature a device has but runs wrong or
// ai: hangs on so costs its speed, not its blocks.
// ai: The outcomes are kept (FormRecord: <cache>/gpu-forms.json) for the device and driver, a variant's held to its
// ai: setup's bytes (whose module names are their sources' hashes), so the trial runs once an install and driver; a
// ai: trial is marked there before it runs, and a mark a start finds (the process ended in a trial: a hang the driver
// ai: never gave back, a crash) refuses that form.
#pragma once
#include "front.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lizard {

using Log = std::function<void(const std::string&)>;

class FormRecord {
 public:
  // ai: dir: the cache folder ("" keeps the record in memory alone, so every start runs the trial)
  FormRecord(const std::string& dir, const wg::Device& d, Log log);
  // ai: a variant's outcomes held to its setup's bytes: another setup forgets them
  void setup(const std::string& variant, const std::vector<uint8_t>& bytes);
  // ai: form "" is the variant itself (refused when no LDPC form of it passes, or when it hangs outside a form's trial)
  bool passed(const std::string& variant, const std::string& form) const;
  bool refused(const std::string& variant, const std::string& form) const;
  void begin(const std::string& variant, const std::string& form);
  void pass(const std::string& variant, const std::string& form);
  void refuse(const std::string& variant, const std::string& form, const std::string& why);
  // ai: a hang or a lost device: the form in trial refused, else the variant
  void lost(const std::string& variant, const std::string& why);
  // ai: a trial that ended in an error of the host's own (no outcome): its mark taken off
  void cancel();
  // ai: every refusal, "variant/form: why", for the log
  std::vector<std::string> refusals() const;

 private:
  void save();
  std::string path_;
  json j_;
  Log log_;
};

// ai: The known frames: luma side x side each, the blocks painted on each (id and payload, XFER_BLOCK bytes a block, in the
// ai: frame's block order).
struct TrialFrames {
  struct Frame { int subch = 0, ring = 0; uint32_t side = 0; std::vector<uint8_t> px; std::vector<std::array<uint8_t, XFER_BLOCK>> blocks; };
  std::vector<Frame> frames;
  uint32_t side = 0;   // ai: the largest
  double paintMs = 0;
};
// ai: The known frames' noise, grey levels (standard deviation): from lizard_gpu_check noise on the 4090 (2026-11-01,
// ai: the 476-byte block's z-128 codes, every frame in its own paint at 2 or more pixels a module; every variant and
// ai: form, 448 blocks a run): every block read from 0 to 32, the first lost at 36, as under the format before; 24 is
// ai: 3.5 dB inside that edge. (With LIZARD-128 painted at a pixel a module and shown at two, the first block lost at 22.)
constexpr double TRIAL_NOISE = 24;
// ai: pictures: the setup's ("pictures", a size each with its top sub-channels)
TrialFrames paintTrial(const json& pictures, double noise = TRIAL_NOISE);

// ai: One decode of the frames in fh's forms as they stand (fh planned for them): ok when every block came back.
struct TrialRun { bool ok = false; int painted = 0, verified = 0, wrong = 0; double its = 0, ms = 0; std::string why; };
TrialRun trialRun(FrontHalf& fh, const TrialFrames& t);

// ai: A device made again after a hang (the receiver's, as it made the first).
using DeviceMaker = std::function<std::unique_ptr<wg::Device>()>;
struct GpuDecoder { std::unique_ptr<wg::Device> dev; std::unique_ptr<FrontHalf> fh; };
// ai: The first of variants (the device's order) whose trial passes, built on d.dev (made by make where null), its
// ai: refused forms left out; false with why where every variant is refused. A hang or a lost device in a trial refuses
// ai: what ran and makes the device again.
bool buildTrialled(GpuDecoder& d, const DeviceMaker& make, ReadFile read, const std::vector<std::string>& variants, double budget, FormRecord& rec, Log log, std::string& why);

}  // namespace lizard
