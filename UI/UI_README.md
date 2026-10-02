## Running the UI
    pip install -r requirements.txt
    streamlit run src/ui/app.py
The app uses a mock backend until `src/search/search.py` (M2) and
`src/detection/detect.py` (M1) are importable; edit the imports in
`src/ui/backend.py` to match their real module and function names.
