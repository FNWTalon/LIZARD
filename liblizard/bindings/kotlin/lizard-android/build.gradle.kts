// ai: The Android packaging (an AAR, 2026-10-03): the library built by the NDK for arm64 with its JNI (the library's
// ai: own CMake), the Kotlin API's sources (../lizard's, compiled for Android: System.loadLibrary finds the AAR's
// ai: liblizard.so), and the engines' GPU files as assets under lizard/ (cmake/gpu_files.cmake's rule, run as a script
// ai: over the generated tree), which LizardAssets.path copies out of the APK once an install. -PlizardVariants=f32-sg,...
// ai: picks the decoder variants (all six by default: 21 MB; f32-sg alone about 10, the sender's files included).
plugins {
  id("com.android.library")
  kotlin("android")
}

val lib: File = rootDir.resolve("../..")
// ai: the generated GPU tree; a tree without it (a fresh clone) still configures, so the JVM library builds and tests,
// ai: and the AAR's assets task says what to run
val gpuTree: File = lib.resolve("out")
val variants = ((findProperty("lizardVariants") as String?) ?: "int8-sg,int8,f16-sg,f16,f32-sg,f32").split(",")

// ai: the files the engines read, by cmake/gpu_files.cmake (the library install's rule) in script mode: the SDK's CMake
// ai: (the one the native build takes, 3.22.1) where it is there, else the PATH's
fun gpuFiles(): List<String> {
  if (!gpuTree.resolve("setup/send.json").isFile)
    throw GradleException("no generated GPU tree in ${gpuTree.normalize()}: lizard-android/build.sh tools, then gen")
  val sdkCmake = androidComponents.sdkComponents.sdkDirectory.get().asFile.resolve("cmake/3.22.1/bin/cmake")
  val list = layout.buildDirectory.file("gpu-files.txt").get().asFile.also { it.parentFile.mkdirs() }
  providers.exec {
    commandLine(if (sdkCmake.canExecute()) sdkCmake.path else "cmake", "-DLIZ_GPU_TREE=${gpuTree.absolutePath}",
      "-DLIZ_GPU_VARIANTS=${variants.joinToString(";")}", "-DLIZ_GPU_LIST=${list.absolutePath}", "-P", lib.resolve("cmake/gpu_files.cmake").absolutePath)
  }.result.get()
  return list.readLines().filter { it.isNotEmpty() }
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
