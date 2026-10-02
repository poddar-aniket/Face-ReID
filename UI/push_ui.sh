#!/usr/bin/env bash
# Run from the root of your local Face-ReID clone, after copying these files in.
set -e
git fetch origin
git checkout -b UI origin/develop 2>/dev/null || git checkout -b UI
mkdir -p src/ui .streamlit
cp -r "$(dirname "$0")/src/ui/." src/ui/
cp "$(dirname "$0")/.streamlit/config.toml" .streamlit/
touch src/__init__.py
cat "$(dirname "$0")/requirements-ui.txt" >> requirements.txt

git add src/ui .streamlit src/__init__.py requirements.txt
git commit -m "feat: add Streamlit UI with mock backend adapter"
git push -u origin UI
