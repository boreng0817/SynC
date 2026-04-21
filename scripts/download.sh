
#!/usr/bin/env bash
set -Eeuo pipefail

links=(
  "https://github.com/boreng0817/SynC/releases/download/v1.0/annotations.z01"
  "https://github.com/boreng0817/SynC/releases/download/v1.0/annotations.zip"
  "https://github.com/boreng0817/SynC/releases/download/v1.0/SHA256SUMS.txt"
)

download_file() {
  local url="$1"
  local name
  name="$(basename "$url")"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$name" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$name"
  else
    echo "Error: wget or curl is required."
    exit 1
  fi
}

echo "[1/4] Downloading release assets..."
for link in "${links[@]}"; do
  download_file "$link"
done

echo "[2/4] Verifying checksums..."
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c SHA256SUMS.txt
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c SHA256SUMS.txt
else
  echo "Error: shasum or sha256sum is required."
  exit 1
fi

echo "[3/4] Merging split zip..."
if ! command -v zip >/dev/null 2>&1; then
  echo "Error: zip command is required for split-zip merge."
  exit 1
fi
zip -s- annotations.zip -O annotations_full.zip

echo "[4/4] Extracting annotations..."
if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip command is required."
  exit 1
fi
unzip -o annotations_full.zip

echo "Cleaning up archive files..."
rm -f annotations.z01 annotations.zip annotations_full.zip SHA256SUMS.txt

echo "Done."
