// ai: The desktop sender (2026-10-03): a native sender with the web sender's UI, painting on the CPU or on Vulkan. Its
// ai: stack, for portability and vsync: Kotlin on the JVM, the UI in Compose Multiplatform,
// ai: the code drawn by the native presenter (lizard-desktop/native/presenter.cpp: its own Vulkan swapchain, FIFO, on the code
// ai: area's AWT canvas, its window found through JAWT), the encoders liblizard/core's (as the Android app's) over JNI
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
// ai: painter's kernels and tables (liblizard/out: setup/send.json, its TAB blob, spv/send_*.spv; ../lizard-android/build.sh gen).
// ai: Native.kt extracts them at start to a folder named by their hash.
val bundled = layout.buildDirectory.dir("bundled")
val liblizardOut = rootDir.resolve("../liblizard/out")
val hostOs = System.getProperty("os.name").lowercase()
val onMac = hostOs.contains("mac")
val nativeLib = rootDir.resolve("build/native/" + when {
    hostOs.contains("win") -> "lizard_desktop.dll"
    onMac -> "liblizard_desktop.dylib"
    else -> "liblizard_desktop.so"
})
// ai: macOS (2026-10-23): MoltenVK, Vulkan over Metal, carried beside the library (its rpath is the library's own
// ai: folder): build/native/libMoltenVK.dylib, from a MoltenVK release (the desktop workflow fetches it)
val moltenVk = rootDir.resolve("build/native/libMoltenVK.dylib")
// ai: the TAB blob send.json names (the paint's codes); a tree without liblizard/out or the native library (a fresh clone) still
// ai: configures, and bundleInputs says what to run (a Sync with no files is skipped, its own actions too)
val sendJson = liblizardOut.resolve("setup/send.json")
val tab = if (sendJson.isFile) Regex("\"blob\"\\s*:\\s*\"([^\"]+)\"").find(sendJson.readText())?.groupValues?.get(1) else null
val bundleInputs by tasks.registering {
    doLast {
        if (!nativeLib.isFile) throw GradleException("no $nativeLib: cmake -S native -B build/native && cmake --build build/native")
        if (onMac && !moltenVk.isFile) throw GradleException("no $moltenVk: libMoltenVK.dylib from a MoltenVK release (MoltenVK/dynamic/dylib/macOS)")
        if (tab == null) throw GradleException("no GPU painter's files in ${liblizardOut.normalize()}: ../lizard-android/build.sh tools, then gen")
    }
}
val bundle by tasks.registering(Sync::class) {
    dependsOn(bundleInputs)
    into(bundled)
    from(nativeLib) { into("lizard/native") }
    if (onMac) from(moltenVk) { into("lizard/native") }
    from(liblizardOut) {
        include(listOfNotNull("setup/send.json", tab, "spv/send_*.spv"))
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
// ai: the type (2026-10-10): the web's faces from liblizard/vendor/fonts, outside the bundle as buildInfo is (Parts.kt
// ai: Inter and Mono read them by their resource paths, fonts/...)
val fonts by tasks.registering(Sync::class) {
    into(layout.buildDirectory.dir("fonts"))
    from(rootDir.resolve("../liblizard/vendor/fonts")) {
        include("inter/InterVariable.ttf", "jetbrains-mono/JetBrainsMono-Regular.ttf")
        into("fonts")
    }
}
sourceSets.main { resources.srcDir(bundle); resources.srcDir(buildInfo); resources.srcDir(fonts); kotlin.srcDir(layout.projectDirectory.dir("../liblizard/kotlin")) }

// ai: after packageDeb: the native library's own Depends added (packaging/deb-deps.sh says which and why)
val debDeps by tasks.registering(Exec::class) {
    commandLine("sh", project.file("packaging/deb-deps.sh").path, layout.buildDirectory.dir("compose/binaries/main/deb").get().asFile.path)
}
tasks.matching { it.name == "packageDeb" }.configureEach { finalizedBy(debDeps) }

// ai: The packages' license text: the repository's LICENSE and NOTICE, then the licenses of the vendored code compiled into
// ai: the native library and of the fonts the window draws in, each under its name (Wirehair's BSD 3-Clause asks for its notice wherever binaries go)
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
        "Skia (in Compose's skiko, which draws the window)", "Inter (liblizard/vendor/fonts/inter), under the SIL Open Font License 1.1",
        "JetBrains Mono (liblizard/vendor/fonts/jetbrains-mono), under the SIL Open Font License 1.1"))
    parts.from(rootDir.resolve("../LICENSE"), rootDir.resolve("../NOTICE"), lib.resolve("vendor/wirehair/LICENSE"),
        lib.resolve("core/third_party/volk/LICENSE.md"), lib.resolve("core/third_party/json.LICENSE.MIT"), lib.resolve("vendor/blake3/LICENSE_A2"),
        project.file("packaging/skia.LICENSE"), lib.resolve("vendor/fonts/inter/LICENSE.txt"), lib.resolve("vendor/fonts/jetbrains-mono/OFL.txt"))
    // ai: macOS: MoltenVK's (Apache 2.0), carried beside the library; its release's LICENSE at build/native
    if (onMac) { titles.add("MoltenVK (Vulkan over Metal, macOS)"); parts.from(rootDir.resolve("build/native/MoltenVK.LICENSE")) }
    out.set(layout.buildDirectory.file("license/copyright.txt"))
}

// ai: The packages' version (2026-10-23): the release's, the Android app's versionCode as the last number (1.0.4 for
// ai: 0.4), since an MSI upgrades only to a higher version and a macOS bundle's must start at 1 or more; the releases'
// ai: files carry the versionName (lizard-android/app/build.gradle.kts, the one place it is set).
val androidGradle = rootDir.resolve("../lizard-android/app/build.gradle.kts").readText()
val releaseCode = Regex("versionCode\\s*=\\s*(\\d+)").find(androidGradle)?.groupValues?.get(1) ?: "0"
val releaseName = Regex("versionName\\s*=\\s*\"([^\"]+)\"").find(androidGradle)?.groupValues?.get(1) ?: "0"
val releaseVersion by tasks.registering {
    doLast { println(releaseName) }
}

// ai: macOS: the bundle signed ad hoc once jpackage has made it (no developer identity): a downloaded app whose
// ai: signature is broken opens as "damaged", one signed ad hoc as an unverified developer's, which System Settings,
// ai: Privacy & Security, lets through
tasks.matching { it.name == "createDistributable" || it.name == "createReleaseDistributable" }.configureEach {
    if (onMac) doLast {
        outputs.files.files.flatMap { d -> d.walkTopDown().maxDepth(2).filter { it.name.endsWith(".app") }.toList() }.forEach { app ->
            val code = ProcessBuilder("codesign", "--force", "--deep", "--sign", "-", app.path).inheritIO().start().waitFor()
            if (code != 0) throw GradleException("codesign of ${app.name} exited $code")
        }
    }
}

// ai: macOS: the license text in the app's resources (Contents/app/resources/LICENSES.txt)
val appLicense by tasks.registering(Copy::class) {
    from(licenseText.flatMap { it.out }) { rename { "LICENSES.txt" } }
    into(layout.buildDirectory.dir("appres/macos"))
}
tasks.matching { it.name == "prepareAppResources" }.configureEach { dependsOn(appLicense) }

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
            // ai: the app's name as a user reads it: LIZARD, in capitals (the launcher bin/LIZARD, the menu entry)
            packageName = "LIZARD"
            packageVersion = "1.0.$releaseCode"
            description = "LIZARD sender"
            vendor = "LIZARD"
            // ai: the repository's license, NOTICE and the vendored code's licenses (licenseText), which jpackage writes
            // ai: into the package (the .deb's share/doc/copyright said "License: Unknown" without them; the MSI shows it
            // ai: as its license page). macOS: inside the app as LICENSES.txt (appLicense), since jpackage makes a
            // ai: license file the .dmg's click-through agreement
            copyright = "Copyright 2026 The LIZARD authors"
            if (onMac) appResourcesRootDir.set(layout.buildDirectory.dir("appres")) else licenseFile.set(licenseText.flatMap { it.out })
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
            // ai: Windows (2026-10-23): an MSI installed for the user alone (no administrator), its folder chosen, a
            // ai: Start menu entry and a desktop shortcut; one upgrade code for every release, so a newer one replaces it
            windows {
                iconFile.set(project.file("packaging/lizard.ico"))
                menuGroup = "LIZARD"
                shortcut = true
                dirChooser = true
                perUserInstall = true
                upgradeUuid = "5b0c1d6e-7f4a-4a7e-9a52-1c2f0e9d6b31"
            }
            // ai: macOS (2026-10-23): Apple Silicon, the mark on Apple's rounded body (packaging/lizard.icns)
            macOS {
                bundleID = "dev.lizard.sender"
                iconFile.set(project.file("packaging/lizard.icns"))
                dockName = "LIZARD"
                appCategory = "public.app-category.utilities"
                minimumSystemVersion = "12.0"
            }
        }
    }
}
