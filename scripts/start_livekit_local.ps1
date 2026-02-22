$ErrorActionPreference = 'Stop'
$server = Join-Path $PSScriptRoot '..\tools\livekit-server\livekit-server.exe'
$server = [System.IO.Path]::GetFullPath($server)
if (-not (Test-Path $server)) {
  Write-Error "livekit-server.exe not found at $server"
}
Write-Host "Starting local LiveKit server in dev mode..."
Write-Host "URL: ws://127.0.0.1:7880"
Write-Host "API Key: devkey"
Write-Host "API Secret: secret"
& $server --dev
