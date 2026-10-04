// ai: The JVM library: the Kotlin API and the natives this machine built (the "jvm" CMake preset's liblizard with its
// ai: JNI) as resources under lizard/native/<os>-<arch>/; its tests load that build directly (lizard.library).
plugins { kotlin("jvm") }

kotlin { jvmToolchain(17) }

dependencies { testImplementation(kotlin("test")) }

val lib = rootDir.resolve("../../build-native/jvm/liblizard.so")
val natives by tasks.registering(Sync::class) {
  from(lib)
  into(layout.buildDirectory.dir("natives/lizard/native/linux-x86_64"))
}
sourceSets.main { resources.srcDir(natives.map { it.destinationDir.parentFile.parentFile.parentFile }) }

tasks.test {
  useJUnitPlatform()
  systemProperty("lizard.library", lib.absolutePath)
  maxHeapSize = "2g"
  // ai: CheckJNI: a JNI call inside a critical region, a bad array access or a pending exception fails the run
  // ai: (2026-10-03: without it the tests passed over a push that called back into Java inside one)
  // ai: HotSpot's CheckJNI only warns of a JNI call inside a critical region (the reverted push printed 110 warnings
  // ai: and every test passed), and writes to the process's own output, below System.out and so outside the test
  // ai: reports: the VM's output is copied to a file, and any warning in it fails the run
  val vmLog = layout.buildDirectory.file("checkjni.log").get().asFile
  jvmArgs("-Xcheck:jni", "-XX:+UnlockDiagnosticVMOptions", "-XX:+LogVMOutput", "-XX:LogFile=${vmLog.absolutePath}")
  doFirst { vmLog.delete() }
  doLast {
    val warned = if (vmLog.exists()) vmLog.readLines().filter { "Calling other JNI functions" in it || "WARNING in native method" in it || "FATAL ERROR in native method" in it } else emptyList()
    if (warned.isNotEmpty()) throw GradleException("CheckJNI: ${warned.size} warnings, the first: ${warned.first().trim()}")
  }
  testLogging { events("passed", "failed"); showStandardStreams = true }
}
