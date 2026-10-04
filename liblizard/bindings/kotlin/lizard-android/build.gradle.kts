// ai: The Android packaging (an AAR, 2026-10-03): the library built by the NDK for arm64 with its JNI (the library's
// ai: own CMake), the Kotlin API's sources (../lizard's, compiled for Android: System.loadLibrary finds the AAR's
// ai: liblizard.so), and the engines' GPU files as assets under lizard/ (cmake/gpu_files.cmake's rule, read here from the
// ai: generated tree), which LizardAssets.path copies out of the APK once an install. -PlizardVariants=f32-sg,... picks
// ai: the decoder variants (all six by default: 21 MB; f32-sg alone about 10, the sender's files included).
import groovy.json.JsonSlurper

plugins {
  id("com.android.library")
  kotlin("android")
}

val lib: File = rootDir.resolve("../..")
// ai: the generated GPU tree; a tree without it (a fresh clone) still configures, so the JVM library builds and tests,
// ai: and the AAR's assets task says what to run
val gpuTree: File = lib.resolve("out")
val variants = ((findProperty("lizardVariants") as String?) ?: "int8-sg,int8,f16-sg,f16,f32-sg,f32").split(",")

@Suppress("UNCHECKED_CAST")
fun gpuFiles(): List<String> {
  if (!gpuTree.resolve("setup/send.json").isFile)
    throw GradleException("no generated GPU tree in ${gpuTree.normalize()}: lizard-android/build.sh tools, then gen")
  val json = JsonSlurper()
  val files = linkedSetOf("setup/send.json", "spv/ingest_camera.spv")
  gpuTree.resolve("spv").listFiles { f -> f.name.startsWith("send_") && f.name.endsWith(".spv") && !f.name.endsWith(".raw.spv") }!!
    .forEach { files += "spv/${it.name}" }
  files += ((json.parse(gpuTree.resolve("setup/send.json")) as Map<String, Any>)["perm"] as Map<String, Any>)["blob"] as String
  for (v in variants) {
    files += "setup/$v.json"
    val objects = (json.parse(gpuTree.resolve("setup/$v.json")) as Map<String, Any>)["objects"] as Map<String, Any>
    for (p in (objects["pipelines"] as Map<String, Map<String, Any>>).values) files += "spv/${p["module"]}.spv"
    for (b in (objects["buffers"] as Map<String, Map<String, Any>>).values) (b["blob"] as? String)?.let { files += "blobs/$it.bin" }
  }
  return files.toList()
}

// ai: the files into <output>/lizard/, registered as every variant's generated assets (the app's LizardAssets way)
abstract class GpuAssets : DefaultTask() {
  @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE) abstract val files: ConfigurableFileCollection
  @get:Internal abstract val tree: DirectoryProperty
  @get:Input abstract val names: ListProperty<String>
  @get:OutputDirectory abstract val output: DirectoryProperty
  @get:Inject abstract val fs: FileSystemOperations
  @TaskAction fun run() {
    fs.sync {
      from(tree) { include(names.get()) }
      into(output.dir("lizard"))
    }
  }
}
val gpu = tasks.register<GpuAssets>("lizardGpuAssets") {
  val list = gpuFiles()
  tree.set(gpuTree)
  names.set(list)
  files.from(list.map { gpuTree.resolve(it) })
}
androidComponents {
  onVariants { v -> v.sources.assets?.addGeneratedSourceDirectory(gpu, GpuAssets::output) }
}

android {
  namespace = "dev.lizard"
  compileSdk = 36
  ndkVersion = "27.1.12297006"
  defaultConfig {
    minSdk = 29
    // ai: the names JNI finds (consumer-rules.pro), kept in an app that shrinks
    consumerProguardFiles("consumer-rules.pro")
    ndk { abiFilters += "arm64-v8a" }
    externalNativeBuild {
      cmake {
        arguments += listOf("-DANDROID_STL=c++_static", "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON", "-DLIZ_TESTS=OFF")
        targets += "lizard"
      }
    }
  }
  externalNativeBuild { cmake { path = lib.resolve("CMakeLists.txt"); version = "3.22.1" } }
  sourceSets["main"].java.srcDirs("../lizard/src/main/kotlin", "src/main/kotlin")
  androidResources { noCompress += listOf("spv", "bin", "json") }
  compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
}
kotlin { jvmToolchain(17) }
