// ai: The presenter's surface on macOS (macos.h): a CAMetalLayer over the code area's canvas, as JAWT's surface layers
// ai: take one: set as the canvas's layer, the JDK adds it to the window's layer and moves it with the canvas's bounds
// ai: from then on (AWTSurfaceLayers' setBounds, from the AWT thread, as here). Built with ARC.
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>

#include <dlfcn.h>
#include <objc/runtime.h>

#include <cstdio>

#include <cmath>

#include <jawt.h>
#include <jawt_md.h>

#include "macos.h"

namespace lizard::mac {

namespace {

// ai: the canvas's surface layers, kept on the layer for its detach
const char kSurfaceLayers = 0;

// ai: DIAGNOSTIC (2026-10-23, the layer not shown on the CI runner): each window's views and layers, from the root
void logLayer(CALayer* l, int depth) {
  std::fprintf(stderr, "lizard: macOS: %*s%s %p at %.0f, %.0f, %.0f x %.0f, z %.0f%s, %lu sub, flipped %d\n", 2 * depth, "",
               object_getClassName(l), (__bridge void*)l, l.frame.origin.x, l.frame.origin.y, l.frame.size.width, l.frame.size.height,
               l.zPosition, l.hidden ? ", hidden" : "", static_cast<unsigned long>(l.sublayers.count), l.geometryFlipped);
  if (depth < 5) for (CALayer* c in l.sublayers) logLayer(c, depth + 1);
}
void logView(NSView* v, int depth) {
  std::fprintf(stderr, "lizard: macOS: %*sview %s at %.0f, %.0f, %.0f x %.0f, layer %s %p, wants %d\n", 2 * depth, "", object_getClassName(v),
               v.frame.origin.x, v.frame.origin.y, v.frame.size.width, v.frame.size.height, v.layer ? object_getClassName(v.layer) : "none",
               (__bridge void*)v.layer, v.wantsLayer);
  if (depth < 4) for (NSView* c in v.subviews) logView(c, depth + 1);
}
void logWindows() {
  for (NSWindow* w in NSApp.windows) {
    std::fprintf(stderr, "lizard: macOS: window \"%s\" visible %d\n", w.title.UTF8String, w.visible);
    NSView* root = w.contentView.superview ? w.contentView.superview : w.contentView;
    logView(root, 1);
    if (w.contentView.layer) logLayer(w.contentView.layer, 1);
  }
}

// ai: the window layer's sublayers on stderr, from the back: what stands over the code where it does not show
void logTree(CALayer* window, const char* when) {
  std::fprintf(stderr, "lizard: macOS: %s, window layer %s, %lu sublayers:\n", when, object_getClassName(window),
               static_cast<unsigned long>(window.sublayers.count));
  for (CALayer* l in window.sublayers)
    std::fprintf(stderr, "lizard: macOS:   %s at %.0f, %.0f, %.0f x %.0f, z %.0f%s%s, opacity %.2f\n", object_getClassName(l),
                 l.frame.origin.x, l.frame.origin.y, l.frame.size.width, l.frame.size.height, l.zPosition, l.hidden ? ", hidden" : "",
                 l.opaque ? ", opaque" : "", l.opacity);
}

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
  id<JAWT_SurfaceLayers> sl = nil;
  CAMetalLayer* layer = nil;
  JAWT_DrawingSurfaceInfo* dsi = ds->GetDrawingSurfaceInfo(ds);
  if (dsi) {
    sl = (__bridge id<JAWT_SurfaceLayers>)dsi->platformInfo;
    if (sl) window = sl.windowLayer;
    if (window) {
      layer = [CAMetalLayer layer];
      layer.opaque = YES;
      layer.backgroundColor = CGColorCreateGenericRGB(1, 0, 0, 1);   // ai: DIAGNOSTIC: red, to see where it is drawn
      objc_setAssociatedObject(layer, &kSurfaceLayers, sl, OBJC_ASSOCIATION_RETAIN);
      // ai: above Compose's own layer (skiko's, a sublayer of the same window layer, over the window's whole content)
      layer.zPosition = 1000;
      [CATransaction begin];
      [CATransaction setDisableActions:YES];
      sl.layer = layer;
      [CATransaction commit];
      // ai: its frame now (the JDK sets it again only when the canvas's bounds next change)
      setFrame(layer, window, p);
    }
    ds->FreeDrawingSurfaceInfo(dsi);
  }
  ds->Unlock(ds);
  awt.FreeDrawingSurface(ds);
  if (!layer) { *err = "the canvas's window has no layer yet"; return nullptr; }
  std::fprintf(stderr, "lizard: macOS: the layer at %.0f, %.0f, %.0f x %.0f points, scale %.2f, in a window layer of %.0f x %.0f\n",
               layer.frame.origin.x, layer.frame.origin.y, layer.frame.size.width, layer.frame.size.height, layer.contentsScale,
               window.bounds.size.width, window.bounds.size.height);
  logTree(window, "at the attach");
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ logTree(window, "2 s later"); logWindows(); });
  return (__bridge_retained void*)layer;
}

void placeLayer(void* handle, const Place& p) {
  if (!handle) return;
  CAMetalLayer* layer = (__bridge CAMetalLayer*)handle;
  setFrame(layer, layer.superlayer, p);
}

void detachLayer(void* handle) {
  if (!handle) return;
  CAMetalLayer* layer = (__bridge_transfer CAMetalLayer*)handle;
  id<JAWT_SurfaceLayers> sl = objc_getAssociatedObject(layer, &kSurfaceLayers);
  [CATransaction begin];
  [CATransaction setDisableActions:YES];
  if (sl && sl.layer == layer) sl.layer = nil;
  else [layer removeFromSuperlayer];
  [CATransaction commit];
  objc_setAssociatedObject(layer, &kSurfaceLayers, nil, OBJC_ASSOCIATION_RETAIN);
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
