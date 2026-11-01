// ai: Strings across JNI as the text they are (2026-10-25, the desktop's; liblizard's since 2026-10-31, the session's
// ai: JNI and the desktop's share it). JNI's own UTF calls (GetStringUTFChars, NewStringUTF) speak modified UTF-8, in
// ai: which a character past U+FFFF (an emoji in a file's name) is two 3-byte surrogates and a 4-byte sequence is not
// ai: valid input, so a path reached the file system mangled and a name read off the light came back wrong or was
// ai: refused. Here a jstring's UTF-16 becomes UTF-8 and back by hand: a surrogate pair is one 4-byte sequence, a lone
// ai: surrogate or bytes that are not UTF-8 become U+FFFD (a maximal invalid subpart one each).
#pragma once
#include <jni.h>

#include <cstddef>
#include <string>

namespace lizard::jni {

// ai: UTF-16 to UTF-8, the conversion str makes (also for a wide environment value on Windows)
std::string utf8(const jchar* s, size_t n);
// ai: a jstring's text as UTF-8 ("" for null)
std::string str(JNIEnv* e, jstring s);
// ai: UTF-8 as a jstring (null only with the VM's OutOfMemoryError pending)
jstring jstr(JNIEnv* e, const std::string& s);

}  // namespace lizard::jni
