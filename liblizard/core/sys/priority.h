// ai: Scheduling priority for the threads that keep frames on time (the sender's present loop, producer, painters and
// ai: uploader; the receiver's camera, phone link and decoder), raised as far as a user's process may, so a busy desktop
// ai: does not drop their frames. LIZ_PRIORITY=normal leaves every priority as the system gives it.
#pragma once
#include <string>

namespace lizard::sys {

// ai: the process's: Windows' priority class HIGH, which every thread runs under (REALTIME, an administrator's, would
// ai: starve the system); elsewhere the calling thread's (raiseThread). What was done, "" nothing.
std::string raiseProcess();

// ai: the calling thread's: Linux nice -20 where the process may (root, CAP_SYS_NICE, RLIMIT_NICE), else the lowest nice
// ai: rtkit grants (RealtimeKit on the system bus, as PipeWire asks for its threads; -15 by default; a thread it raises
// ai: does not pass it on to the threads it starts, so each asks for itself); macOS the user-interactive QoS class;
// ai: Windows and Android nothing beyond the process's (android.cpp sets its camera thread's own). What was done, "".
std::string raiseThread();

}  // namespace lizard::sys
