param(
    [string]$InputPath = "video\this.mp4",
    [string]$Ip = "127.0.0.1",
    [int]$Port = 5000
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvCandidates = @(
    (Join-Path $ProjectRoot "..\SGS\Scripts\python.exe"),
    (Join-Path $ProjectRoot "SGS\Scripts\python.exe")
)

$PythonExe = $VenvCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $PythonExe) {
    Write-Error "SGS virtual environment was not found. Expected ..\SGS or .\SGS from the project root."
    exit 1
}

$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot "Ultralytics"
$env:YOLOV5_CONFIG_DIR = Join-Path $ProjectRoot "Ultralytics"

Set-Location $ProjectRoot
& $PythonExe "main.py" "--Input" $InputPath "--ip" $Ip "--port" $Port
