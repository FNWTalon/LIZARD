/* ai: The JDK's Windows jni_md.h, its definitions as Windows JDKs ship them (the ABI: jint a 32-bit long, jlong 64
 * ai: bits, exports by __declspec), for cross-building the desktop sender's JNI library from Linux with MinGW, where
 * ai: no Windows JDK is installed (2026-10-04). The rest of jni.h and jawt.h is the build JDK's own. */
#ifndef LIZ_WIN32_JNI_MD_H
#define LIZ_WIN32_JNI_MD_H
#define JNIEXPORT __declspec(dllexport)
#define JNIIMPORT __declspec(dllimport)
#define JNICALL __stdcall
typedef long jint;
typedef __int64 jlong;
typedef signed char jbyte;
#endif
