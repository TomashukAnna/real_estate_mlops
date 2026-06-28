param(
    [switch]$UseExistingImages
)

$ErrorActionPreference = "Stop"

$Namespace = "real-estate-mlops"
$Overlay = "k8s/overlays/docker-desktop"
$PortForwardMarker = "real-estate-mlops-port-forward"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Action
}

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Stop-PortForwardProcesses {
    param([array]$Processes)

    foreach ($Process in $Processes) {
        if ($null -ne $Process -and -not $Process.HasExited) {
            Write-Host "Stopping port-forward process $($Process.Id)"
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

Assert-Command "docker"
Assert-Command "kubectl"

Invoke-Step "Checking Kubernetes cluster" {
    kubectl cluster-info | Out-Null
}

if (-not $UseExistingImages) {
    Invoke-Step "Building local Docker images" {
        docker build -t real-estate-api:local .
        docker build -t real-estate-bff:local -f Dockerfile.bff .
        docker build -t real-estate-ui:local ./ui
    }
}

Invoke-Step "Resetting Kubernetes namespace and local PVs" {
    kubectl delete namespace $Namespace --ignore-not-found=true

    try {
        kubectl wait --for=delete "namespace/$Namespace" --timeout=120s
    }
    catch {
        Write-Host "Namespace '$Namespace' is already absent or finished deleting." -ForegroundColor DarkGray
    }

    kubectl delete pv `
        real-estate-data-pv `
        real-estate-models-pv `
        real-estate-reports-pv `
        real-estate-grafana-dashboards-pv `
        --ignore-not-found=true
}

Invoke-Step "Applying Kubernetes manifests" {
    kubectl apply -k $Overlay
}

Invoke-Step "Waiting for deployments" {
    kubectl wait --for=condition=available deployment --all -n $Namespace --timeout=180s
}

Invoke-Step "Current cluster state" {
    kubectl get pods -n $Namespace -o wide
    kubectl get svc -n $Namespace
    kubectl get pvc -n $Namespace
}

Invoke-Step "Stopping existing port-forward processes for this namespace" {
    Get-CimInstance Win32_Process -Filter "Name = 'kubectl.exe'" |
        Where-Object {
            $_.CommandLine -match "port-forward" -and
            $_.CommandLine -match $Namespace
        } |
        ForEach-Object {
            Write-Host "Stopping kubectl port-forward process $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force
        }

    Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object {
            $_.CommandLine -match $PortForwardMarker
        } |
        ForEach-Object {
            Write-Host "Closing port-forward PowerShell window $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force
        }
}

$PortForwards = @(
    @{ Name = "API"; Service = "real-estate-api"; LocalPort = 8000; RemotePort = 8000; Url = "http://localhost:8000" },
    @{ Name = "BFF"; Service = "real-estate-bff"; LocalPort = 8002; RemotePort = 8002; Url = "http://localhost:8002" },
    @{ Name = "UI"; Service = "real-estate-ui"; LocalPort = 8080; RemotePort = 8501; Url = "http://localhost:8080" },
    @{ Name = "Prometheus"; Service = "real-estate-prometheus"; LocalPort = 9090; RemotePort = 9090; Url = "http://localhost:9090" },
    @{ Name = "Grafana"; Service = "real-estate-grafana"; LocalPort = 3000; RemotePort = 3000; Url = "http://localhost:3000" }
)

$ForwardProcesses = @()
try {
    Write-Host ""
    Write-Host "==> Starting port-forward processes" -ForegroundColor Cyan
    foreach ($Forward in $PortForwards) {
        $Arguments = @(
            "port-forward",
            "-n", $Namespace,
            "svc/$($Forward.Service)",
            "$($Forward.LocalPort):$($Forward.RemotePort)"
        )
        $Process = Start-Process "kubectl" `
            -ArgumentList $Arguments `
            -WindowStyle Hidden `
            -PassThru
        $ForwardProcesses += $Process
    }

    Write-Host ""
    Write-Host "Services:" -ForegroundColor Green
    foreach ($Forward in $PortForwards) {
        Write-Host ("  {0,-10} {1}" -f $Forward.Name, $Forward.Url)
    }
    Write-Host "  OpenAPI    http://localhost:8000/docs"
    Write-Host "  Grafana    admin/admin"

    Write-Host ""
    Write-Host "Kubernetes stack is ready. Keep this window open while using forwarded ports." -ForegroundColor Green
    Write-Host "Press Ctrl+C or close this window to stop port-forward processes."

    while ($true) {
        Start-Sleep -Seconds 2
        $ExitedProcess = $ForwardProcesses | Where-Object { $_.HasExited } | Select-Object -First 1
        if ($null -ne $ExitedProcess) {
            throw "Port-forward process $($ExitedProcess.Id) exited unexpectedly."
        }
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping port-forward processes..." -ForegroundColor Cyan
    Stop-PortForwardProcesses -Processes $ForwardProcesses
}
