import os
import sys
import urllib.request
import json
import zipfile
import tarfile
import stat

ENGINE_DIR = os.path.join(os.path.dirname(__file__), ".bin")

def _get_download_url():
    """Fetches the latest release from GitHub API dynamically."""
    api_url = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            release_data = json.loads(response.read().decode())
    except Exception as e:
        raise RuntimeError(f"Could not connect to GitHub API: {e}")

    system = sys.platform.lower()
    assets = release_data.get("assets", [])
    
    download_url = None
    file_name = None

    for asset in assets:
        name = asset["name"].lower()
        if "win" in system and "windows" in name and "avx2" in name and name.endswith(".zip"):
            download_url = asset["browser_download_url"]
            file_name = asset["name"]
            break
        elif "linux" in system and ("ubuntu" in name or "linux" in name) and "avx2" in name and (name.endswith(".tar.gz") or name.endswith(".tar")):
            download_url = asset["browser_download_url"]
            file_name = asset["name"]
            break
        elif "darwin" in system and "macos" in name and (name.endswith(".tar.gz") or name.endswith(".tar")):
            download_url = asset["browser_download_url"]
            file_name = asset["name"]
            break

    if not download_url:
        raise RuntimeError("Could not find a suitable Stockfish binary for your OS.")

    return download_url, file_name

def _is_valid_binary(filename):
    """Helper to distinguish the real engine executable from documentation files."""
    name = filename.lower()
    if "stockfish" not in name:
        return False
    if sys.platform.lower().startswith("win"):
        return name.endswith(".exe")
    return "." not in name

def ensure_stockfish():
    """Downloads, extracts, and configures Stockfish automatically."""
    os.makedirs(ENGINE_DIR, exist_ok=True)
    
    for root, dirs, files in os.walk(ENGINE_DIR):
        for file in files:
            if _is_valid_binary(file):
                return os.path.join(root, file)

    url, archive_name = _get_download_url()
    print(f"[*] Downloading latest Stockfish automatically: {archive_name}")
    archive_path = os.path.join(ENGINE_DIR, archive_name)

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
        out_file.write(response.read())

    print("[*] Extracting files...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(ENGINE_DIR)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tar"):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(ENGINE_DIR)

    os.remove(archive_path)

    final_exe_path = None
    for root, dirs, files in os.walk(ENGINE_DIR):
        for file in files:
            if _is_valid_binary(file):
                final_exe_path = os.path.join(root, file)
                break
        
        if final_exe_path:
            break

    if not final_exe_path:
        raise FileNotFoundError("Critical error: Stockfish binary not found after extraction.")


    if "win" not in sys.platform.lower():
        st = os.stat(final_exe_path)
        os.chmod(final_exe_path, st.st_mode | stat.S_IEXEC)

    print("[*] Stockfish configured successfully!")
    return final_exe_path

STOCKFISH_PATH = ensure_stockfish()