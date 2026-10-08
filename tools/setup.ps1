# Setter opp Python-miljøet for dybdekartene.
#   .\tools\setup.ps1          # CUDA (NVIDIA-GPU), ~5 GB
#   .\tools\setup.ps1 -Cpu     # bare CPU, ~1 GB – tregere, men fungerer
param([switch]$Cpu)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

# depth-anything-3 0.1.1 krever Python <= 3.12
if (-not (Test-Path $python)) { py -3.12 -m venv $venv }
& $python -m pip install --upgrade pip

$index = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu130' }
& $python -m pip install --no-cache-dir torch torchvision --index-url $index
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
& $python -m pip install --no-deps depth-anything-3==0.1.1

& $python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
& $python -c "from depth_anything_3.api import DepthAnything3; print('Depth Anything 3 OK')"
