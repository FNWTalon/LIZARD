# ai: What liblizard's JNI layer finds by name (api/jni.cpp: JNI_OnLoad's FindClass and RegisterNatives, the receiver's
# ai: callbacks' GetMethodID), so an app that shrinks with R8 keeps it (2026-10-03: without these
# ai: rules JNI_OnLoad returned JNI_ERR in any minified app).
-keep class dev.lizard.Native { native <methods>; }
-keep class dev.lizard.LizardException { <init>(int, java.lang.String); }
-keep interface dev.lizard.Receiver$Release { *; }
-keep interface dev.lizard.Receiver$Log { *; }
