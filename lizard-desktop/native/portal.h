// ai: The XDG desktop portal's file chooser (portal.cpp): the desktop's own dialog for files or folders.
#pragma once
#include <string>
#include <vector>

namespace lizard {

// ai: answered: the dialog closed (uris what was chosen: file:// URIs from the portal, paths from Windows' dialog; empty
// ai: when cancelled); else error says why it could not be asked (no sd-bus, no session bus, no portal), for the
// ai: caller's fallback
struct PortalPick {
  bool answered = false;
  std::vector<std::string> uris;
  std::string error;
};

// ai: parent: the window the dialog belongs to, "x11:<hex id>" or "win32:<hex HWND>", or "" for none. Blocks until the
// ai: dialog closes: not on a UI thread.
PortalPick portalOpen(const std::string& parent, const std::string& title, bool directory, bool multiple);

}  // namespace lizard
