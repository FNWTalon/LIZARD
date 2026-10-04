package dev.lizard.receiver

import android.content.Context
import java.io.File

// ai: The generated tree (liblizard/out's setup, blobs, spv: the app's assets/lizard/, app/build.gradle.kts LizardAssets)
// ai: copied to filesDir/lizard once an install, since the native side reads it by path (ReceiverConfig.assets). The
// ai: stamp is the package's lastUpdateTime: every install copies afresh, so a new gen never meets an old tree.
object Assets {
    fun ensure(ctx: Context): File {
        val dir = File(ctx.filesDir, "lizard")
        val stamp = File(dir, ".stamp")
        val want = ctx.packageManager.getPackageInfo(ctx.packageName, 0).lastUpdateTime.toString()
        if (stamp.isFile && stamp.readText() == want) return dir
        dir.deleteRecursively()
        copy(ctx, "lizard", dir)
        stamp.writeText(want)   // ai: last: a copy cut short is redone
        return dir
    }

    private fun copy(ctx: Context, from: String, to: File) {
        val kids = ctx.assets.list(from).orEmpty()
        if (kids.isEmpty()) {
            to.parentFile?.mkdirs()
            ctx.assets.open(from).use { i -> to.outputStream().use { o -> i.copyTo(o, 1 shl 16) } }
            return
        }
        to.mkdirs()
        for (k in kids) copy(ctx, "$from/$k", File(to, k))
    }
}
