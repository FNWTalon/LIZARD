/* ai: The JDK's Windows jawt_md.h, the drawing surface's Windows information as AWT fills it (the ABI), for the MinGW
 * ai: cross build (win32/jni_md.h says why). The presenter reads hwnd alone. */
#ifndef LIZ_WIN32_JAWT_MD_H
#define LIZ_WIN32_JAWT_MD_H
#include <windows.h>
#include "jawt.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct jawt_Win32DrawingSurfaceInfo {
  union {
    HWND hwnd;
    HBITMAP hbitmap;
    void* pbits;
  };
  HDC hdc;
  HPALETTE hpalette;
} JAWT_Win32DrawingSurfaceInfo;
#ifdef __cplusplus
}
#endif
#endif
