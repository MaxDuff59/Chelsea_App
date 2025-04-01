import os

BASE_DIR = "/Users/maxenceduffuler/Desktop/CHELSEA_PROJECT/CODE/STREAMLIT/"

CONFIG = {
    "paths": {
        "gps_data": os.path.join(BASE_DIR, "DATA", "CFC GPS Data.csv"),
        "photo_club": os.path.join(BASE_DIR, "images", "LOGO_TEAMS.xlsx"),
        "individual_priority": os.path.join(BASE_DIR, "DATA", "CFC Individual Priority Areas.csv"),
        "physical_data": os.path.join(BASE_DIR, "DATA", "CFC Physical Capability Data_.csv"),
        "recovery_data": os.path.join(BASE_DIR, "DATA", "CFC Recovery status Data.csv"),
        "icons": os.path.join(BASE_DIR, "icons"),
        "target_data": os.path.join(BASE_DIR, "DATA", "Fake Target GPS Data.csv"),
        "predicted_hr_data": os.path.join(BASE_DIR, "DATA", "Predicted HR Performance.csv"),
        "RPE_data": os.path.join(BASE_DIR, "DATA", "Fake RPE GPS PHYSICAL.csv"),
        "WEIGHT_data": os.path.join(BASE_DIR, "DATA", "Fake Weight Data.csv"),
        "Injury_History": os.path.join(BASE_DIR, "DATA", "Fake Injury History.csv")
    },
    "chart_colors": {
        "primary": "#00148b",
        "secondary": "darkred",
        "highlight": "gold",
        "background": "#0e1117"
    },
    "date_format": "%d/%m/%Y",
    "sliding_window_size": 12
}