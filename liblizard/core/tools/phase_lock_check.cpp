// ai: The phase lock (rx/phase_lock.h) replayed from a trace of its host's calls, every decision and every field of its
// ai: state written out per call, so the lock can be held to another implementation fed the same trace (the Kotlin
// ai: PhaseLock.kt it was ported from) or to an earlier run of its own. The trace: one call a line, tokens apart by
// ai: one space, `#` lines skipped:
// ai:   mode track | mode off                 set
// ai:   reopen                                reopened
// ai:   capture <ts ns> <delayed 0|1> <skew ns> [<us>]   onCapture (a trailing us: the recording lock's own answer,
// ai:                                         not read)
// ai:   fate <taken 0|1>                      delayFate
// ai:   series <count> <v>...                 Receiver::series's vector (the held version, 8 values a frame), through
// ai:                                         framesOf to onWindow
// ai:   window <version> <n> (<ts> <blocks> <found> <r> <rSd> <r2> <r2Sd> <row>) x n   onWindow with frames as given
// ai:   line <ts ns>                          line
// ai: Doubles as Java writes and reads them (NaN, Infinity). The output, a line or more a call, its index i first:
// ai:   i Y <text>                            each log line the call said, in order
// ai:   i E <kind> <answer>                   capture: the us returned; line: its text, or null; the rest: -
// ai:   i S <state>                           state(), where it changed: null, or arm, what (spaces as _), a, b, k,
// ai:                                         se, n, gain, pace, stood, edge, flatMs, delays, refreshMs, multiple,
// ai:                                         frameMs, landMs, row
// ai:   i H <hash>                            FNV-1a 64 of every field of the lock (the I line), every call
// ai:   i I <fields>                          (with --full) those fields by name
// ai: A double is its IEEE bits in hex (nan for any NaN), so equal output is equal to the bit.
// ai: --host drives the lock through its host (rx/phase_host.h) instead, from a trace of the Android app's own order (a
// ai: Save replay's inputs: the mode read every 30 captures, the feed, the line every 300): its mode and line calls are
// ai: the host's own and left alone, each series call is what the receiver answers the host's next ask, and the delays
// ai: it asks are all taken; it writes the projection (--project: a line `C <us>` a capture, `S <state>` where it
// ai: changed), which the plain run's projection of the same trace must equal.
//
//   phase_lock_check <trace> [out] [--full] [--debug] [--project | --host]   out: the output file (stdout without one);
//                                       --debug: the lock's debug lines too; exit 1 on a trace it cannot read (--host:
//                                       or one whose feed the host does not ask as the trace's did)
#include "phase_host.h"
#include "phase_lock.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace lizard {

// ai: every field of the lock, in the Kotlin runner's order (PhaseLockCheck.kt), as the I line
struct PhaseLockProbe {
  static std::string hex(double x) {
    if (std::isnan(x)) return "nan";
    uint64_t u;
    std::memcpy(&u, &x, 8);
    char b[24];
    std::snprintf(b, sizeof b, "%016" PRIx64, u);
    return b;
  }
  static std::string fields(const PhaseLock& p) {
    std::ostringstream o;
    auto L = [&](const char* k, int64_t v) { o << k << '=' << v << ' '; };
    auto D = [&](const char* k, double v) { o << k << '=' << hex(v) << ' '; };
    auto B = [&](const char* k, bool v) { o << k << '=' << (v ? 1 : 0) << ' '; };
    o << "mode=" << (p.mode_ == PhaseLock::Mode::Track ? "track" : "off") << ' ';
    L("delays", p.delays);
    L("delaysAll", p.delaysAll);
    L("first", p.first);
    L("now", p.now);
    L("seen", p.seen);
    B("wasDelayed", p.wasDelayed);
    L("skew", p.skew);
    L("plains", (int64_t)p.plains.size());
    L("plainsLast", p.plains.empty() ? 0 : p.plains.back()[1]);
    L("frame", p.frame);
    L("owed", p.owed);
    B("flagged", p.flagged);
    L("flagAt", p.flagAt);
    B("longPrev", p.longPrev);
    L("askT", p.askT);
    L("lands", (int64_t)p.lands.size());
    L("landsLast", p.lands.empty() ? 0 : p.lands.back()[1]);
    L("base", p.base);
    D("pace", p.pace);
    D("paceVar", p.paceVar);
    L("paceAt", p.paceAt);
    L("multiple", p.multiple);
    D("row", p.row);
    L("settle", p.settle);
    L("line", p.line_);
    B("move", p.move);
    B("moving", p.moving);
    L("window", (int64_t)p.window.size());
    L("windowLast", p.window.empty() ? 0 : p.window.back().ts);
    L("since", p.since);
    D("level", p.level);
    L("placed", (int64_t)p.placed.size());
    L("placedLast", p.placed.empty() ? 0 : p.placed.rbegin()->second[0]);
    D("gain", p.gain);
    D("gainVar", p.gainVar);
    D("gpCov", p.gpCov);
    D("prevE", p.prevE);
    D("prevSe", p.prevSe);
    L("prevT", p.prevT);
    D("prevHop", p.prevHop);
    D("prevPace", p.prevPace);
    D("prevX", p.prevX);
    L("prevSide", p.prevSide);
    L("evid", (int64_t)p.evid.size());
    D("evidLast", p.evid.empty() ? 0.0 : p.evid.back()[3]);
    L("aimFrom", p.aimFrom);
    B("balanced", p.balanced);
    if (p.lastLeak) {
      const auto& e = *p.lastLeak;
      o << "leak=" << hex(e.a) << ',' << hex(e.b) << ',' << hex(e.sa) << ',' << hex(e.sb) << ',' << e.n << ',' << e.t << ',' << hex(e.k) << ','
        << hex(e.x) << ' ';
    } else {
      o << "leak=null ";
    }
    L("edgeDir", p.edgeDir);
    L("edgeIn", p.edgeIn);
    L("edgeT", p.edgeT);
    B("edgeClean", p.edgeClean);
    L("flatHalf", p.flatHalf);
    B("stood", p.stood);
    std::string s = o.str();
    if (!s.empty()) s.pop_back();
    return s;
  }
};

}  // namespace lizard

using lizard::PhaseLock;
using lizard::PhaseLockProbe;

static uint64_t fnv(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

static std::string stateLine(const std::optional<PhaseLock::State>& s) {
  if (!s) return "null";
  std::string what = s->what;
  for (char& c : what)
    if (c == ' ') c = '_';
  std::ostringstream o;
  auto D = [&](double v) { o << ' ' << PhaseLockProbe::hex(v); };
  o << s->arm << ' ' << what;
  D(s->a);
  D(s->b);
  D(s->k);
  D(s->se);
  o << ' ' << s->n;
  D(s->gain);
  D(s->pace);
  o << ' ' << (s->stood ? 1 : 0) << ' ' << s->edge;
  D(s->flatMs);
  o << ' ' << s->delays;
  D(s->refreshMs);
  o << ' ' << s->multiple;
  D(s->frameMs);
  D(s->landMs);
  D(s->row);
  return o.str();
}

static double num(const std::string& t) { return std::strtod(t.c_str(), nullptr); }
static int64_t lng(const std::string& t) { return std::strtoll(t.c_str(), nullptr, 10); }

static std::vector<std::string> tokens(const std::string& l) {
  std::vector<std::string> t;
  std::istringstream ss(l);
  for (std::string w; ss >> w;) t.push_back(w);
  return t;
}

// ai: --host: the trace's calls as the host's inputs (the header)
static int hostRun(std::istream& in, std::ostream& out, const std::string& trace) {
  std::deque<std::vector<double>> answers;   // ai: the series the receiver gives the host's next asks
  double fed = -1.0;                         // ai: the newest ms the trace's feed handed over
  bool apart = false;
  std::string why;
  int64_t asked = 0;   // ai: the delay the host asked of the camera at the capture just given
  lizard::PhaseLockHost h(
      [](const std::string&) {},
      [&](double since) -> std::vector<double> {
        if (answers.empty()) return {0.0};
        std::vector<double> a = answers.front();
        answers.pop_front();
        if (since != fed && !apart) {
          apart = true;
          why = "the host asked after " + std::to_string(since) + " ms, the trace's feed after " + std::to_string(fed);
        }
        if (a.size() > 1) fed = a[1 + 8 * ((a.size() - 1) / 8 - 1)];
        return a;
      },
      [&](int64_t us) {
        asked = us;
        return true;
      });
  h.opened("");
  std::string lastState = "null";
  std::string l;
  while (std::getline(in, l)) {
    if (l.empty() || l[0] == '#') continue;
    const std::vector<std::string> t = tokens(l);
    if (t[0] == "series") {
      const size_t n = (size_t)lng(t[1]);
      std::vector<double> a(n);
      for (size_t j = 0; j < n && 2 + j < t.size(); j++) a[j] = num(t[2 + j]);
      answers.push_back(a);
    } else if (t[0] == "reopen") {
      h.opened("");
      fed = -1.0;
    } else if (t[0] == "fate") {
      h.fate(t[1] == "1");
    } else if (t[0] == "capture") {
      asked = 0;
      h.captured(lng(t[1]), t[2] == "1");
      if (!answers.empty() && !apart) {
        apart = true;
        why = "a series the host did not ask for";
      }
      out << "C " << asked << '\n';
      const std::string s = stateLine(h.state());
      if (s != lastState) {
        out << "S " << s << '\n';
        lastState = s;
      }
    }   // ai: (mode and line: the host's own)
  }
  out.flush();
  if (apart) {
    std::fprintf(stderr, "phase_lock_check --host: %s: %s\n", trace.c_str(), why.c_str());
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  std::string trace, outPath;
  bool full = false, debug = false, project = false, host = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--full") {
      full = true;
    } else if (a == "--debug") {
      debug = true;
    } else if (a == "--project") {
      project = true;
    } else if (a == "--host") {
      host = true;
    } else if (trace.empty()) {
      trace = a;
    } else {
      outPath = a;
    }
  }
  if (trace.empty()) {
    std::fprintf(stderr, "phase_lock_check <trace> [out] [--full] [--debug] [--project | --host]\n");
    return 2;
  }
  std::ifstream in(trace);
  if (!in) {
    std::fprintf(stderr, "phase_lock_check: cannot read %s\n", trace.c_str());
    return 1;
  }
  std::ofstream outFile;
  if (!outPath.empty()) outFile.open(outPath);
  std::ostream& out = outPath.empty() ? std::cout : outFile;
  if (host) return hostRun(in, out, trace);
  long long i = 0;
  PhaseLock p(
      [&](const std::string& s) {
        if (!project) out << i << " Y " << s << '\n';
      },
      debug);
  std::string lastState = "null";
  std::string l;
  std::vector<std::string> t;
  long long lineNo = 0;
  while (std::getline(in, l)) {
    lineNo++;
    if (l.empty() || l[0] == '#') continue;
    t.clear();
    std::istringstream ss(l);
    for (std::string w; ss >> w;) t.push_back(w);
    const std::string& k = t[0];
    std::string answer = "-";
    auto need = [&](size_t n) {
      if (t.size() < n) {
        std::fprintf(stderr, "phase_lock_check: %s line %lld: %s wants %zu tokens\n", trace.c_str(), lineNo, k.c_str(), n);
        std::exit(1);
      }
    };
    if (k == "mode") {
      need(2);
      p.set(t[1] == "track" ? PhaseLock::Mode::Track : PhaseLock::Mode::Off);
    } else if (k == "reopen") {
      p.reopened();
    } else if (k == "capture") {
      need(4);
      answer = std::to_string(p.onCapture(lng(t[1]), t[2] == "1", lng(t[3])));
    } else if (k == "fate") {
      need(2);
      p.delayFate(t[1] == "1");
    } else if (k == "series") {
      need(2);
      const size_t n = (size_t)lng(t[1]);
      need(2 + n);
      std::vector<double> a(n);
      for (size_t j = 0; j < n; j++) a[j] = num(t[2 + j]);
      int version = 0;
      const std::vector<PhaseLock::Frame> fs = PhaseLock::framesOf(a, version);
      p.onWindow(fs, version);
    } else if (k == "window") {
      need(3);
      const int version = (int)lng(t[1]);
      const size_t n = (size_t)lng(t[2]);
      need(3 + 8 * n);
      std::vector<PhaseLock::Frame> fs(n);
      for (size_t j = 0; j < n; j++) {
        const size_t o = 3 + 8 * j;
        fs[j].ts = lng(t[o]);
        fs[j].blocks = (int)lng(t[o + 1]);
        fs[j].found = t[o + 2] == "1";
        fs[j].r = num(t[o + 3]);
        fs[j].rSd = num(t[o + 4]);
        fs[j].r2 = num(t[o + 5]);
        fs[j].r2Sd = num(t[o + 6]);
        fs[j].row = num(t[o + 7]);
      }
      p.onWindow(fs, version);
    } else if (k == "line") {
      need(2);
      const std::optional<std::string> s = p.line(lng(t[1]));
      answer = s ? *s : "null";
    } else {
      std::fprintf(stderr, "phase_lock_check: %s line %lld: no call %s\n", trace.c_str(), lineNo, k.c_str());
      return 1;
    }
    const std::string st = stateLine(p.state());
    if (project) {
      if (k == "capture") {
        out << "C " << answer << '\n';
        if (st != lastState) {
          out << "S " << st << '\n';
          lastState = st;
        }
      }
      i++;
      continue;
    }
    out << i << " E " << k << ' ' << answer << '\n';
    if (st != lastState) {
      out << i << " S " << st << '\n';
      lastState = st;
    }
    const std::string f = PhaseLockProbe::fields(p);
    char h[24];
    std::snprintf(h, sizeof h, "%016" PRIx64, fnv(f));
    out << i << " H " << h << '\n';
    if (full) out << i << " I " << f << '\n';
    i++;
  }
  out.flush();
  return 0;
}
