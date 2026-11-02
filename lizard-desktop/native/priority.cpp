// ai: The app's priority raised at launch (Native.raisePriority from Main.kt, before the window's threads start;
// ai: LIZ_PRIORITY=normal leaves it): ../../liblizard/core/sys/priority.h, Windows' priority class HIGH, elsewhere the
// ai: calling thread's; the threads that keep frames on time raise their own as they start. What was done, "" nothing.
#include <jni.h>

#include "priority.h"

extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_desktop_Native_raisePriority(JNIEnv* env, jclass) {
  return env->NewStringUTF(lizard::sys::raiseProcess().c_str());
}
