// ai: liblizard's Kotlin binding: lizard (the JVM library, desktops and Android alike) and lizard-android (the AAR: the
// ai: same sources, the library built by the NDK, the GPU files as assets).
pluginManagement {
  repositories { gradlePluginPortal(); mavenCentral(); google() }
}
dependencyResolutionManagement {
  repositories { mavenCentral(); google() }
}
rootProject.name = "lizard-kotlin"
include(":lizard", ":lizard-android")
