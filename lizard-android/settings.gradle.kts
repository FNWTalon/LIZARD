// ai: The Android app's Gradle project: one module, app/. liblizard/core is reached from
// ai: the app's CMake (app/src/main/cpp/CMakeLists.txt), liblizard/gen and out from lizard-android/build.sh, not from Gradle.
pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "lizard"
include(":app")
