// ai: The apps' shared UI (2026-10-31): the look of LIZARD's screens in Compose Multiplatform, written once and compiled
// ai: into the Android app (lizard-android, Jetpack Compose under the Android Gradle plugin) and the desktop app
// ai: (lizard-desktop, Compose Desktop). It has no build of its own: each app's settings include it as ":lizard-ui", so the
// ai: plugins' versions are the including app's (its root build file names them), and its outputs go under that app's
// ai: build folder (build/lizard-ui), never shared between the two builds.
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("multiplatform")
    id("com.android.library")
    id("org.jetbrains.compose")
    id("org.jetbrains.kotlin.plugin.compose")
}

layout.buildDirectory.set(rootProject.layout.buildDirectory.dir("lizard-ui"))

kotlin {
    jvmToolchain(17)
    androidTarget { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }
    jvm("desktop")
    sourceSets {
        commonMain {
            // ai: liblizard/kotlin's plain-JVM code both apps use (Readout.kt, the screens' words and figures; Pick.kt, the
            // ai: format a room holds; Tar.kt, a bundle's tar), compiled here once for both; PhaseLock.kt, the reference
            // ai: the C++ phase lock is held to, left out. commonMain sees the JDK: its two targets are both JVMs.
            kotlin.srcDir("../liblizard/kotlin")
            kotlin.exclude("PhaseLock.kt")
            dependencies {
                api(compose.runtime)
                api(compose.foundation)
                api(compose.material3)
                implementation(compose.components.resources)
                // ai: the native side's JSON as both apps read it (State.kt; its element API alone: no serialization plugin)
                api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
            }
        }
    }
}

android {
    namespace = "dev.lizard.ui"
    compileSdk = 36
    defaultConfig { minSdk = 29 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// ai: The type (2026-10-10): the web's faces from liblizard/vendor/fonts (Inter's variable file, read at each weight by
// ai: its axis; JetBrains Mono Regular), as this module's font resources (Res.font), named as resources must be
val fonts by tasks.registering(Sync::class) {
    val vendor = layout.projectDirectory.dir("../liblizard/vendor/fonts")
    from(vendor.file("inter/InterVariable.ttf")) { rename { "inter_variable.ttf" } }
    from(vendor.file("jetbrains-mono/JetBrainsMono-Regular.ttf")) { rename { "jetbrains_mono_regular.ttf" } }
    into(layout.buildDirectory.dir("fonts/font"))
}

compose.resources {
    packageOfResClass = "dev.lizard.ui.res"
    customDirectory("commonMain", fonts.map { layout.buildDirectory.dir("fonts").get() })
}
