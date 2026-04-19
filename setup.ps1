$vcpkgRoot = "$HOME\vcpkg"
if (!(Test-Path $vcpkgRoot)) {
    git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
    & "$vcpkgRoot\bootstrap-vcpkg.bat"
}
$env:VCPKG_ROOT = $vcpkgRoot

$installPath = "$vcpkgRoot\installed\arm64-windows"
$versionHeader = "$installPath\include\opencv4\opencv2\core\version.hpp"

if (!(Test-Path $versionHeader)) {
    Write-Host "OpenCV not found or still building in background. Installing/Ensuring..." -ForegroundColor Yellow
    & "$vcpkgRoot\vcpkg" install opencv4[world,opencl]:arm64-windows --recurse
}

if (Test-Path $versionHeader) {
    Write-Host "OpenCV detected at $installPath" -ForegroundColor Green
    
    $env:OPENCV_LINK_PATHS = "$installPath\lib"
    $env:OPENCV_INCLUDE_PATHS = "$installPath\include\opencv4"
    $env:OPENCV_LINK_LIBS = "opencv_world4"
    $env:VCPKGRS_DYNAMIC = "1"
    $env:VCPKGRS_TRIPLET = "arm64-windows"
    $env:PATH = "$installPath\bin;" + $env:PATH
    $env:ORT_STRATEGY = "manual"
    $env:OPENCV_DISABLE_PROBES = "cmake,pkg_config"

    # Runtime Dependencies
    $binDir = "$PSScriptRoot\bin"
    if (!(Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir }
    $env:PATH = "$binDir;" + $env:PATH

    if (!(Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
        Write-Host "Downloading yt-dlp..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe" -OutFile "$binDir\yt-dlp.exe"
    }
    
    if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Write-Host "Downloading ffmpeg (this may take a minute)..." -ForegroundColor Cyan
        $zip = "$binDir\ffmpeg.zip"
        Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath "$binDir\tmp" -Force
        $exe = Get-ChildItem -Recurse -Path "$binDir\tmp" -Filter "ffmpeg.exe" | Select-Object -First 1
        Move-Item $exe.FullName "$binDir\ffmpeg.exe" -Force
        Remove-Item "$binDir\tmp" -Recurse -Force
        Remove-Item $zip -Force
    }

    Write-Host "Environment configured successfully."
} else {
    Write-Error "OpenCV installation failed."
}
