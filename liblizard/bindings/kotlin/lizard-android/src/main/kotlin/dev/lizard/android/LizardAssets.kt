// ai: The engines' GPU files on Android (2026-10-03): the AAR carries them as assets under lizard/ (its build.gradle.kts),
// ai: and the native side reads them by path, so they are copied to filesDir/lizard once an install (the app's own
// ai: Assets.kt scheme: the stamp is the package's lastUpdateTime, so a new build never meets an old tree). The path
// ai: is what Receiver's and Sender's assets take.
package dev.lizard.android

import android.content.Context
import java.io.File

object LizardAssets {
  fun path(context: Context): String {
    val dir = File(context.filesDir, "lizard")
    val stamp = File(dir, ".stamp")
    val want = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
    if (stamp.isFile && stamp.readText() == want) return dir.path
    dir.deleteRecursively()
    copy(context, "lizard", dir)
    stamp.writeText(want)   // ai: last: a copy cut short is redone
    return dir.path
  }

  private fun copy(context: Context, from: String, to: File) {
    val kids = context.assets.list(from).orEmpty()
    if (kids.isEmpty()) {
      to.parentFile?.mkdirs()
      context.assets.open(from).use { i -> to.outputStream().use { o -> i.copyTo(o, 1 shl 16) } }
      return
    }
    to.mkdirs()
    for (k in kids) copy(context, "$from/$k", File(to, k))
  }
}
