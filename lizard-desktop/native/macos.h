// ai: The presenter's surface on macOS (macos.mm): a CAMetalLayer of its own laid over the code area's canvas, added to
// ai: the AWT window's layer tree through JAWT's surface layers, and presented by MoltenVK (VK_EXT_metal_surface). AWT
// ai: on macOS gives a component no native view, so the layer is placed by hand: at the canvas's bounds in the window's
// ai: content, in points, its pixels the window's backing scale.
#pragma once
#include <jni.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lizard::mac {

// ai: where the canvas lies, in points from the top left of the window's content view, and the backing scale
struct Place { double x = 0, y = 0, w = 0, h = 0, scale = 1; };

// ai: a new CAMetalLayer at p, retained, added to the canvas's window through JAWT on the main thread; null, with why in
// ai: err, where JAWT gives no surface layers. On the calling thread JAWT is held only to read the window's layer.
void* attachLayer(JNIEnv* e, jobject canvas, const Place& p, std::string* err);
// ai: the layer moved and sized to p (on the main thread, no animation)
void placeLayer(void* layer, const Place& p);
// ai: the layer out of its window and released (on the main thread)
void detachLayer(void* layer);
// ai: the layer's size in pixels (its bounds by its contentsScale), for a surface that leaves the extent to the swapchain
bool layerPixels(void* layer, uint32_t* w, uint32_t* h);
// ai: the refresh of the screen holding the layer's window (NSScreen's maximumFramesPerSecond), 0 unknown
double layerRefreshHz(void* layer);

// ai: VSync (presenter.cpp holdMode), on the main display as layerRefreshHz reads it: its id; the refreshes of its modes
// ai: at its size in pixels and points, usable for the desktop, with the refresh of its mode now in *now (0 where the
// ai: mode does not say: a built-in panel); that size's mode at hz set for this app alone (kCGConfigureForAppOnly: the
// ai: system undoes it at the app's exit); every display back at its own (CGRestorePermanentDisplayConfiguration)
uint32_t mainDisplay();
std::vector<double> displayRates(uint32_t display, double* now);
bool setDisplayRate(uint32_t display, double hz);
void restoreDisplays();

}  // namespace lizard::mac
