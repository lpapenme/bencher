set -eu

for version_file in /opt/bencher/*/.python-version; do
  version=$(tr -d '[:space:]' < "$version_file")
  python="${version_file%/.python-version}/.venv/bin/python"
  actual=$("$python" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
  [ "$actual" = "$version" ] || {
    echo "$python is Python $actual; expected $version"
    exit 1
  }
  resolved=$(readlink -f "$python")
  case "$resolved" in
    /opt/uv-python/*) ;;
    *) echo "$python resolves outside /opt/uv-python: $resolved"; exit 1 ;;
  esac
done
