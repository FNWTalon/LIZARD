pluginManagement {
    repositories { gradlePluginPortal(); mavenCentral(); google() }
}
dependencyResolutionManagement {
    repositories { mavenCentral(); google() }
}
rootProject.name = "lizard-sender"
// ai: the apps' shared UI (Compose Multiplatform, ../lizard-ui), built as part of this build; the Android app includes it alike
include(":lizard-ui")
project(":lizard-ui").projectDir = file("../lizard-ui")
