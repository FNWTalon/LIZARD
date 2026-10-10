// ai: The receiver app: Kotlin, Compose, Camera2, and libliz.so (src/main/cpp) over liblizard/core/.
// ai: core's lizard_rx by default; -PlizStub=true links the stub receiver (receiver_stub.cpp) instead.
import javax.inject.Inject
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val lizStub = (findProperty("lizStub") as String?)?.toBoolean() ?: false

android {
    namespace = "dev.lizard.receiver"
    compileSdk = 36
    ndkVersion = "27.1.12297006"
    defaultConfig {
        applicationId = "dev.lizard.receiver"
        minSdk = 29
        targetSdk = 36
        versionCode = 2
        versionName = "0.2"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                // ai: flexible page sizes: the NDK 27 switch that links for 16 KB pages (Android 15's devices)
                arguments += listOf("-DANDROID_STL=c++_static", "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
                    "-DLIZ_STUB_RECEIVER=${if (lizStub) "ON" else "OFF"}")
                targets += "liz"
            }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            // ai: a lab build: release signed with the debug key so it installs without a keystore
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true }
    // ai: the setup's tables, blobs and SPIR-V stored, not deflated: copied out once, and a stored entry streams
    androidResources { noCompress += listOf("spv", "bin", "json") }
    // ai: native libs uncompressed and 16 KB aligned in the APK
    packaging { jniLibs { useLegacyPackaging = false } }
}

kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }

// ai: liblizard/out's setup, blobs and device SPIR-V (not naga's .raw.spv, not wgsl/, naga/, spvx/) as assets/lizard/;
// ai: MainActivity copies them to filesDir/lizard/ on the first run after an install (Assets.kt).
abstract class LizardAssets : DefaultTask() {
    @get:Internal abstract val out: DirectoryProperty
    @get:OutputDirectory abstract val output: DirectoryProperty
    @get:Inject abstract val fs: FileSystemOperations

    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE)
    val picked: FileTree get() = out.asFileTree.matching { PICK(this) }

    @TaskAction fun run() {
        // ai: an APK without its GPU files builds and then fails at run time ("The camera could not start: lizard")
        val tree = out.get().asFile
        if (!tree.resolve("setup/send.json").isFile) throw GradleException("no generated GPU tree in ${tree.normalize()}: lizard-android/build.sh tools, then gen")
        fs.sync {
            from(out) { PICK(this) }
            into(output.dir("lizard"))
        }
    }

    companion object {
        val PICK: (PatternFilterable) -> Unit = {
            it.include("setup/**", "blobs/**", "spv/*.spv", "nets/**", "gen.json")
            it.exclude("spv/*.raw.spv")
        }
    }
}

val lizardAssets = tasks.register<LizardAssets>("lizardAssets") {
    out.set(rootProject.layout.projectDirectory.dir("../liblizard/out"))
}
androidComponents {
    onVariants { v -> v.sources.assets?.addGeneratedSourceDirectory(lizardAssets, LizardAssets::output) }
}

dependencies {
    val bom = platform("androidx.compose:compose-bom:2025.06.00")
    implementation(bom)
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
}
