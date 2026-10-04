// ai: The desktop sender (2026-10-03): a native sender with the web sender's UI, painting on the CPU or on Vulkan. Its
// ai: stack, for portability and vsync: Kotlin on the JVM, the UI in Compose Multiplatform,
// ai: the code drawn by the native presenter (lizard-desktop/native/presenter.cpp: its own Vulkan swapchain, FIFO, on the code
// ai: area's AWT canvas, its window found through JAWT), the encoders the Android app's native core over JNI
// ai: (lizard-desktop/native).
import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import java.time.Instant
import java.time.temporal.ChronoUnit

plugins {
    kotlin("jvm") version "2.1.20"
    id("org.jetbrains.kotlin.plugin.compose") version "2.1.20"
    id("org.jetbrains.compose") version "1.8.2"
}

kotlin { jvmToolchain(17) }

dependencies {
    implementation(compose.desktop.currentOs)
    implementation(compose.material3)
    // ai: the stats JSON the native side writes (its element API alone: no serialization plugin)
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
}

// ai: What the app carries as resources: the native library (lizard-desktop/build/native, CMake on native/) and the GPU
// ai: painter's kernels and tables (liblizard/out: setup/send.json, its PERMW blob, spv/send_*.spv; ../lizard-android/build.sh gen).
// ai: Native.kt extracts them at start to a folder named by their hash.
val bundled = layout.buildDirectory.dir("bundled")
val liblizardOut = rootDir.resolve("../liblizard/out")
val nativeLib = rootDir.resolve("build/native/" + System.getProperty("os.name").lowercase().let { os ->
    when {
        os.contains("win") -> "lizard_desktop.dll"
        os.contains("mac") -> "liblizard_desktop.dylib"
        else -> "liblizard_desktop.so"
    }
})
// ai: the PERMW blob send.json names; a tree without liblizard/out or the native library (a fresh clone) still
// ai: configures, and bundleInputs says what to run (a Sync with no files is skipped, its own actions too)
val sendJson = liblizardOut.resolve("setup/send.json")
val perm = if (sendJson.isFile) Regex("\"blob\"\\s*:\\s*\"([^\"]+)\"").find(sendJson.readText())?.groupValues?.get(1) else null
val bundleInputs by tasks.registering {
    doLast {
        if (!nativeLib.isFile) throw GradleException("no $nativeLib: cmake -S native -B build/native && cmake --build build/native")
        if (perm == null) throw GradleException("no GPU painter's files in ${liblizardOut.normalize()}: ../lizard-android/build.sh tools, then gen")
    }
}
val bundle by tasks.registering(Sync::class) {
    dependsOn(bundleInputs)
    into(bundled)
    from(nativeLib) { into("lizard/native") }
    from(liblizardOut) {
        include(listOfNotNull("setup/send.json", perm, "spv/send_*.spv"))
        exclude("spv/*.raw.spv")
        into("lizard/assets")
    }
}
// ai: the files' list, which Native.kt reads (a jar's folders cannot be listed)
bundle.configure {
    doLast {
        val root = bundled.get().asFile.resolve("lizard")
        root.resolve("files.txt").writeText(root.walkTopDown().filter { it.isFile && it.name != "files.txt" }
            .joinToString("\n") { it.relativeTo(root).invariantSeparatorsPath })
    }
}
// ai: the About line's build time (the web's about(): "Build 2026-10-04 12:00 UTC"), outside the bundle so the native
// ai: files' hash, and the folder Native.kt extracts them to, stays the same from one build to the next
val buildInfo by tasks.registering {
    val dir = layout.buildDirectory.dir("buildinfo")
    outputs.dir(dir)
    outputs.upToDateWhen { false }
    doLast {
        dir.get().asFile.resolve("lizard-build.txt").apply { parentFile.mkdirs() }
            .writeText(Instant.now().truncatedTo(ChronoUnit.SECONDS).toString())
    }
}
sourceSets.main { resources.srcDir(bundle); resources.srcDir(buildInfo) }

// ai: after packageDeb: the native library's own Depends added (packaging/deb-deps.sh says which and why)
val debDeps by tasks.registering(Exec::class) {
    commandLine("sh", project.file("packaging/deb-deps.sh").path, layout.buildDirectory.dir("compose/binaries/main/deb").get().asFile.path)
}
tasks.matching { it.name == "packageDeb" }.configureEach { finalizedBy(debDeps) }

// ai: The packages' license text: the repository's LICENSE and NOTICE, then the licenses of the vendored code compiled into
// ai: the native library, each under its name (Wirehair's BSD 3-Clause asks for its notice wherever binaries go)
abstract class LicenseText : DefaultTask() {
    @get:Input abstract val titles: ListProperty<String>
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val parts: ConfigurableFileCollection
    @get:OutputFile abstract val out: RegularFileProperty
    @TaskAction fun write() {
        val rule = "\n\n" + "-".repeat(78) + "\n\n"
        out.get().asFile.writeText(titles.get().zip(parts.files.toList()).joinToString(rule) { (t, f) ->
            (if (t.isEmpty()) "" else "$t\n\n") + f.readText().trimEnd() } + "\n")
    }
}
val licenseText by tasks.registering(LicenseText::class) {
    val lib = rootDir.resolve("../liblizard")
    titles.set(listOf("", "", "Wirehair (liblizard/vendor/wirehair)", "volk (liblizard/core/third_party/volk)",
        "JSON for Modern C++ (liblizard/core/third_party/json.hpp)", "BLAKE3 (liblizard/vendor/blake3), under its Apache License 2.0 option",
        "Skia (in Compose's skiko, which draws the window)"))
    parts.from(rootDir.resolve("../LICENSE"), rootDir.resolve("../NOTICE"), lib.resolve("vendor/wirehair/LICENSE"),
        lib.resolve("core/third_party/volk/LICENSE.md"), lib.resolve("core/third_party/json.LICENSE.MIT"), lib.resolve("vendor/blake3/LICENSE_A2"),
        project.file("packaging/skia.LICENSE"))
    out.set(layout.buildDirectory.file("license/copyright.txt"))
}

compose.desktop {
    application {
        mainClass = "dev.lizard.desktop.MainKt"
        jvmArgs += listOf("-Dsun.java2d.uiScale.enabled=true")
        // ai: The packages (2026-10-04): jpackage puts a Java runtime
        // ai: in each, jlink's cut of the build JDK to the modules the app uses (suggestRuntimeModules: java.prefs the kept
        // ai: settings, jdk.unsupported Compose's Skia, java.instrument), so nothing needs Java installed.
        // ai: `./gradlew packageDeb` (debDeps after it adds the native library's own needs to Depends).
        nativeDistributions {
            targetFormats(TargetFormat.Deb, TargetFormat.Msi, TargetFormat.Dmg)
            packageName = "Lizard"
            packageVersion = "1.0.0"
            description = "Lizard sender"
            vendor = "Lizard"
            // ai: the repository's license, NOTICE and the vendored code's licenses (licenseText), which jpackage writes
            // ai: into the package (the .deb's share/doc/copyright said "License: Unknown" without them)
            copyright = "Copyright 2026 The Lizard authors"
            licenseFile.set(licenseText.flatMap { it.out })
            modules("java.instrument", "java.prefs", "jdk.unsupported")
            linux {
                packageName = "lizard-sender"
                // ai: the project's public address (GitHub's private one for its account), never a person's email
                debMaintainer = "262231965+FNWTalon@users.noreply.github.com"
                iconFile.set(project.file("packaging/lizard.png"))
                shortcut = true
                menuGroup = "Utility"
                appCategory = "utils"
            }
        }
    }
}
