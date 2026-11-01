// ai: The Android app's Gradle project: app/, and the shared UI (:lizard-ui, ../lizard-ui). liblizard/core is reached from
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
// ai: the apps' shared UI (Compose Multiplatform, ../lizard-ui), built as part of this build; the desktop app includes it alike
include(":lizard-ui")
project(":lizard-ui").projectDir = file("../lizard-ui")
