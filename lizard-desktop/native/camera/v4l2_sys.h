// ai: The calls the V4L2 backend (v4l2.cpp) makes (2026-10-25), as a table a test stands a fake in for
// ai: (sim/fake_v4l2.h): the device nodes to probe, open, close, ioctl, mmap, munmap, poll, and the /dev/v4l/by-id link
// ai: naming a node ("" none). Each call keeps the system's convention: -1 (MAP_FAILED for mmap) with errno set.
// ai: The kernel's table: nodes() is probedNodes over LIZ_V4L2_NODES and /dev/video* sorted by number; the other calls
// ai: are the system's. Opening goes through nodes() too (v4l2Open finds the id among them), so a camera no probe
// ai: lists is never opened.
#pragma once
#include <poll.h>
#include <sys/types.h>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace lizard::cam {

// ai: The nodes a probe asks (2026-10-25): where env (LIZ_V4L2_NODES's value) is set, the nodes it names, colon-
// ai: separated, empty parts skipped, so set and empty is none, as every test that lists cameras sets it (no real
// ai: camera reached); unset, what byDefault finds (the kernel's table: every /dev/video* node, so the app lists the
// ai: system's cameras). byDefault is called only where env is unset, so a check passes one of its own and never reads
// ai: /dev.
std::vector<std::string> probedNodes(const char* env, const std::function<std::vector<std::string>()>& byDefault);

struct V4l2Sys {
  std::function<std::vector<std::string>()> nodes;
  std::function<int(const char* path, int flags)> open;
  std::function<int(int fd)> close;
  std::function<int(int fd, unsigned long request, void* arg)> ioctl;
  std::function<void*(void* addr, size_t length, int prot, int flags, int fd, off_t offset)> mmap;
  std::function<int(void* addr, size_t length)> munmap;
  std::function<int(pollfd* fds, nfds_t n, int timeoutMs)> poll;
  std::function<std::string(const std::string& node)> byId;
};

// ai: The table in use: the kernel's unless one was swapped in. A camera keeps the table it was opened through.
const V4l2Sys& v4l2Sys();
// ai: Swaps the table in use (null: the kernel's) and returns the one before (never null: the kernel's where it was).
// ai: A table swapped in outlives every camera opened through it.
const V4l2Sys* v4l2UseSys(const V4l2Sys* table);

}  // namespace lizard::cam
