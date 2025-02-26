# Script d'installation de l'environnement IA sous Windows

# Vérifier si Python est installé
$pythonVersion = python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python non trouvé. Installation en cours..."
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe" -OutFile "python-installer.exe"
    Start-Process -FilePath "python-installer.exe" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
    Remove-Item "python-installer.exe"
} else {
    Write-Host "Python déjà installé : $pythonVersion"
}

# Vérifier que pip est bien installé et mis à jour
python -m ensurepip
python -m pip install --upgrade pip

# Créer un environnement virtuel
Write-Host "Création de l'environnement virtuel (llm_env)..."
python -m venv llm_env
Write-Host "Environnement virtuel créé."

# Activer l’environnement
Write-Host "Activation de l'environnement..."
& llm_env\Scripts\activate

# Vérifier si CUDA est installé (uniquement si NVIDIA est présent)
if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") {
    Write-Host "CUDA détecté, installation de PyTorch avec support GPU..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "CUDA non détecté, installation de PyTorch CPU..."
    pip install torch torchvision torchaudio
}

# Installer les dépendances IA
Write-Host "Installation des librairies IA..."
pip install transformers datasets peft bitsandbytes accelerate scipy numpy pandas tqdm

# Tester l'installation
Write-Host "Vérification de l'installation..."
python -c "import torch; print('PyTorch fonctionne avec GPU ? ', torch.cuda.is_available())"

Write-Host "Installation terminée !"
