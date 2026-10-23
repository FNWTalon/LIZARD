// ai: The presenter's surface on macOS (macos.h): a CAMetalLayer over the code area's canvas, through JAWT's surface
// ai: layers. Built with ARC. Every change to the layer tree runs on the main thread (AppKit's), dispatched without
// ai: waiting: the AWT thread that calls in may be one the main thread waits on.
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>

#include <dlfcn.h>

#include <cmath>

#include <jawt.h>
#include <jawt_md.h>

#include "macos.h"

namespace lizard::mac {

namespace {

// ai: the frame in the window layer's own coordinates: AWT's are from the top left, the layer's from the bottom left
void setFrame(CAMetalLayer* l, CALayer* window, const Place& p) {
  const CGFloat H = window ? window.bounds.size.height : p.y + p.h;
  [CATransaction begin];
  [CATransaction setDisableActions:YES];
  l.contentsScale = p.scale > 0 ? p.scale : 1;
  l.frame = CGRectMake(p.x, H - p.y - p.h, p.w, p.h);
  [CATransaction commit];
}

}  // namespace

void* attachLayer(JNIEnv* e, jobject canvas, const Place& p, std::string* err) {
  // ai: JAWT_GetAWT from the jawt the app loaded before this library (Native.kt), looked up rather than linked, as on
  // ai: Windows: the JDK names it @rpath/libjawt.dylib, which this library's rpath does not reach
  using GetAWT = jboolean(JNICALL*)(JNIEnv*, JAWT*);
  const auto getAwt = reinterpret_cast<GetAWT>(dlsym(RTLD_DEFAULT, "JAWT_GetAWT"));
  JAWT awt;
  // ai: 9 alone: macOS's JAWT takes the layer mode as given from 1.7 on, and refuses 9 with JAWT_MACOSX_USE_CALAYER
  awt.version = JAWT_VERSION_9;
  if (!getAwt || getAwt(e, &awt) == JNI_FALSE) { *err = "JAWT is not available"; return nullptr; }
  JAWT_DrawingSurface* ds = awt.GetDrawingSurface(e, canvas);
  if (!ds) { *err = "the canvas has no drawing surface"; return nullptr; }
  const jint lock = ds->Lock(ds);
  if (lock & JAWT_LOCK_ERROR) { awt.FreeDrawingSurface(ds); *err = "the canvas's drawing surface could not be locked"; return nullptr; }
  CALayer* window = nil;
  JAWT_DrawingSurfaceInfo* dsi = ds->GetDrawingSurfaceInfo(ds);
  if (dsi) {
    id<JAWT_SurfaceLayers> sl = (__bridge id<JAWT_SurfaceLayers>)dsi->platformInfo;
    if (sl) window = sl.windowLayer;
    ds->FreeDrawingSurfaceInfo(dsi);
  }
  ds->Unlock(ds);
  awt.FreeDrawingSurface(ds);
  if (!window) { *err = "the canvas's window has no layer yet"; return nullptr; }
  CAMetalLayer* layer = [CAMetalLayer layer];
  layer.opaque = YES;
  layer.backgroundColor = CGColorGetConstantColor(kCGColorWhite);
  // ai: sized before the swapchain is made on the presenter's thread; the window's height read again on the main thread
  setFrame(layer, window, p);
  dispatch_async(dispatch_get_main_queue(), ^{
    [window addSublayer:layer];
    setFrame(layer, window, p);
  });
  return (__bridge_retained void*)layer;
}

void placeLayer(void* handle, const Place& p) {
  if (!handle) return;
  CAMetalLayer* layer = (__bridge CAMetalLayer*)handle;
  dispatch_async(dispatch_get_main_queue(), ^{ setFrame(layer, layer.superlayer, p); });
}

void detachLayer(void* handle) {
  if (!handle) return;
  CAMetalLayer* layer = (__bridge_transfer CAMetalLayer*)handle;
  dispatch_async(dispatch_get_main_queue(), ^{
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [layer removeFromSuperlayer];
    [CATransaction commit];
  });
}

bool layerPixels(void* handle, uint32_t* w, uint32_t* h) {
  if (!handle) return false;
  CAMetalLayer* layer = (__bridge CAMetalLayer*)handle;
  const CGSize s = layer.bounds.size;
  const CGFloat k = layer.contentsScale;
  *w = static_cast<uint32_t>(s.width * k + 0.5);
  *h = static_cast<uint32_t>(s.height * k + 0.5);
  return true;
}

double layerRefreshHz(void*) {
  // ai: the main display's mode (CoreGraphics answers on any thread; a built-in panel may say 0, and the presenter's
  // ai: measurement corrects either)
  CGDisplayModeRef m = CGDisplayCopyDisplayMode(CGMainDisplayID());
  if (!m) return 0;
  const double hz = CGDisplayModeGetRefreshRate(m);
  CGDisplayModeRelease(m);
  return hz;
}

uint32_t mainDisplay() { return CGMainDisplayID(); }

namespace {
// ai: the display's modes, the duplicate low-resolution ones listed too (a refresh may live only there); CFRelease it
CFArrayRef allModes(CGDirectDisplayID d) {
  const void* keys[] = {kCGDisplayShowDuplicateLowResolutionModes};
  const void* vals[] = {kCFBooleanTrue};
  CFDictionaryRef opts = CFDictionaryCreate(nullptr, keys, vals, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFArrayRef all = CGDisplayCopyAllDisplayModes(d, opts);
  if (opts) CFRelease(opts);
  return all;
}
bool sameSize(CGDisplayModeRef a, CGDisplayModeRef b) {
  return CGDisplayModeGetPixelWidth(a) == CGDisplayModeGetPixelWidth(b) && CGDisplayModeGetPixelHeight(a) == CGDisplayModeGetPixelHeight(b) &&
         CGDisplayModeGetWidth(a) == CGDisplayModeGetWidth(b) && CGDisplayModeGetHeight(a) == CGDisplayModeGetHeight(b);
}
}  // namespace

std::vector<double> displayRates(uint32_t display, double* now) {
  std::vector<double> hz;
  *now = 0;
  CGDisplayModeRef cur = CGDisplayCopyDisplayMode(display);
  if (!cur) return hz;
  *now = CGDisplayModeGetRefreshRate(cur);
  if (CFArrayRef all = allModes(display)) {
    for (CFIndex i = 0; i < CFArrayGetCount(all); i++) {
      auto m = static_cast<CGDisplayModeRef>(const_cast<void*>(CFArrayGetValueAtIndex(all, i)));
      const double r = CGDisplayModeGetRefreshRate(m);
      if (r > 0 && CGDisplayModeIsUsableForDesktopGUI(m) && sameSize(m, cur)) hz.push_back(r);
    }
    CFRelease(all);
  }
  CGDisplayModeRelease(cur);
  return hz;
}

bool setDisplayRate(uint32_t display, double hz) {
  CGDisplayModeRef cur = CGDisplayCopyDisplayMode(display);
  if (!cur) return false;
  bool ok = false;
  if (CFArrayRef all = allModes(display)) {
    for (CFIndex i = 0; i < CFArrayGetCount(all) && !ok; i++) {
      auto m = static_cast<CGDisplayModeRef>(const_cast<void*>(CFArrayGetValueAtIndex(all, i)));
      if (std::fabs(CGDisplayModeGetRefreshRate(m) - hz) >= 0.01 || !CGDisplayModeIsUsableForDesktopGUI(m) || !sameSize(m, cur)) continue;
      CGDisplayConfigRef cfg = nullptr;
      if (CGBeginDisplayConfiguration(&cfg) != kCGErrorSuccess) break;
      if (CGConfigureDisplayWithDisplayMode(cfg, display, m, nullptr) != kCGErrorSuccess) { CGCancelDisplayConfiguration(cfg); break; }
      ok = CGCompleteDisplayConfiguration(cfg, kCGConfigureForAppOnly) == kCGErrorSuccess;
    }
    CFRelease(all);
  }
  CGDisplayModeRelease(cur);
  return ok;
}

void restoreDisplays() { CGRestorePermanentDisplayConfiguration(); }

}  // namespace lizard::mac
