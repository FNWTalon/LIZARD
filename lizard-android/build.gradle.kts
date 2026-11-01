plugins {
    id("com.android.application") version "8.11.0" apply false
    id("org.jetbrains.kotlin.android") version "2.1.20" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.1.20" apply false
    // ai: the shared UI's (../lizard-ui): a Kotlin Multiplatform library, its Android target and its desktop one
    id("com.android.library") version "8.11.0" apply false
    id("org.jetbrains.kotlin.multiplatform") version "2.1.20" apply false
    id("org.jetbrains.compose") version "1.8.2" apply false
}
