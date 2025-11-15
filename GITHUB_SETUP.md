# GitHub Repository Setup Guide

## Voraussetzungen

1. ✅ Git LFS ist installiert und initialisiert
2. ✅ `.gitattributes` wurde erstellt
3. ✅ `.gitignore` wurde angepasst
4. ⚠️ GitHub CLI Login erforderlich

## Schritt 1: GitHub Login

Führe folgenden Befehl aus, um dich bei GitHub anzumelden:

```powershell
gh auth login
```

Folge den Anweisungen:
- Wähle "GitHub.com"
- Wähle "HTTPS" oder "SSH"
- Authentifiziere dich (Browser oder Token)

## Schritt 2: Repository erstellen und pushen

### Option A: Automatisch mit Script (Empfohlen)

```powershell
.\scripts\setup-github-repo.ps1 -GitHubUser "DEIN-GITHUB-USERNAME" -RepoName "Bimify"
```

Oder für ein privates Repository:

```powershell
.\scripts\setup-github-repo.ps1 -GitHubUser "DEIN-GITHUB-USERNAME" -RepoName "Bimify" -Private
```

### Option B: Manuell

1. **Erstelle das Repository auf GitHub:**
   - Gehe zu https://github.com/new
   - Repository Name: `Bimify`
   - Wähle Public oder Private
   - **WICHTIG:** Erstelle KEIN README, keine .gitignore, keine License (wir haben diese bereits)

2. **Füge den Remote hinzu:**
   ```powershell
   git remote add origin https://github.com/DEIN-USERNAME/Bimify.git
   ```

3. **Stage alle Dateien:**
   ```powershell
   git add .
   ```

4. **Erstelle den ersten Commit:**
   ```powershell
   git commit -m "Initial commit: Bimify IFC Export V2 with Git LFS"
   ```

5. **Setze den Branch (falls nötig):**
   ```powershell
   git branch -M main
   ```

6. **Pushe zu GitHub:**
   ```powershell
   git push -u origin main
   ```

## Schritt 3: Verifizierung

Nach dem Push, überprüfe:

```powershell
# Zeige alle mit Git LFS getrackten Dateien
git lfs ls-files

# Prüfe den Remote
git remote -v

# Prüfe den Status
git status
```

## Wichtige Hinweise

- **Große Dateien** (IFC, PNG, WASM, PDF, GeoJSON) werden automatisch mit Git LFS getrackt
- **Runtime-Daten** (data/uploads/, data/exports/, data/jobs/) werden ignoriert
- **Beispiel-Dateien** in `examples/` werden mit LFS getrackt, wenn sie groß sind

## Troubleshooting

### "Git LFS not found"
Installiere Git LFS von: https://git-lfs.github.com/

### "Authentication failed"
Führe `gh auth login` erneut aus

### "Repository already exists"
Entweder:
- Lösche das Repository auf GitHub.com und erstelle es neu
- Oder verwende einen anderen Namen: `-RepoName "Bimify-New"`

### "Large file push failed"
Stelle sicher, dass Git LFS korrekt installiert ist:
```powershell
git lfs install
git lfs track "*.ifc"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

