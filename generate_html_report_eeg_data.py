import os
import pandas as pd
from datetime import datetime
import json

# Set working directory
os.chdir("L:\\Lab_LucaC\\Carina")

# Define the main directory containing patient folders
main_eeg_directory = "L:\\Lab_LucaC\\A_QNC_ANT_Data\\TMS_MDD_EEG_data"

# Dictionary to store patient data
patient_data = {}

# Define the cutoff date for "Eyes Closed" sessions
cutoff_date = datetime.strptime("2025-02-11", "%Y-%m-%d")

# Traverse each patient's folder
for patient_folder in sorted(os.listdir(main_eeg_directory)):
    patient_path = os.path.join(main_eeg_directory, patient_folder)
    
    # Check if it's a directory
    if os.path.isdir(patient_path):
        vhdr_files = []
        
        # Get all .vhdr files and their creation times (excluding "renamed" files)
        for file in sorted(os.listdir(patient_path)):  # Sort by ID
            if file.endswith(".vhdr") and "renamed" not in file.lower():  # Exclude renamed files
                file_path = os.path.join(patient_path, file)
                file_time = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d')
                vhdr_files.append((file, file_time))
        
        # Store in dictionary
        patient_data[patient_folder] = sorted(vhdr_files, key=lambda x: x[1])  # Sort by time

# Create summary statistics
session_counts = [len(vhdr) for vhdr in patient_data.values()]
summary_stats = pd.Series(session_counts).value_counts().sort_index()

# Collect all session dates for the timeline
all_dates = []
for sessions in patient_data.values():
    for _, session_date in sessions:
        all_dates.append(session_date)

# Sort dates for timeline visualization
all_dates.sort()

# Generate HTML report
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Session Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { width: 100%%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .summary { margin-top: 20px; }
        .chart-container { width: 100%%; height: 400px; margin-top: 20px; }
        .highlight { color: cyan; font-weight: bold; }
    </style>
</head>
<body>
    <h2>EEG Session Report</h2>
    <p>Last Updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>

    <h3>Summary Statistics</h3>
    <table class="summary">
        <tr>
            <th>Number of Sessions</th>
            <th>Number of Patients</th>
        </tr>
"""

# Add summary statistics
for session_count, num_patients in summary_stats.items():
    html_content += f"""
        <tr>
            <td>{session_count}</td>
            <td>{num_patients}</td>
        </tr>
    """

html_content += """
    </table>

    <h3>EEG Collection Timeline</h3>
    <div class="chart-container">
        <canvas id="timelineChart"></canvas>
    </div>

    <script>
        const ctx = document.getElementById('timelineChart').getContext('2d');
        const sessionDates = """ + json.dumps(all_dates) + """;

        const counts = {};
        sessionDates.forEach(date => {
            counts[date] = (counts[date] || 0) + 1;
        });

        const labels = Object.keys(counts);
        const data = Object.values(counts);

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of EEG Sessions per Day',
                    data: data,
                    borderColor: 'blue',
                    backgroundColor: 'rgba(0, 0, 255, 0.2)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Date' } },
                    y: { title: { display: true, text: 'EEG Sessions' }, beginAtZero: true }
                }
            }
        });
    </script>

    <h3>Patient EEG Sessions</h3>
    <table>
        <tr>
            <th>Patient ID</th>
            <th>Number of Sessions</th>
            <th>Session Dates</th>
        </tr>
"""
# Add patient data with "Eyes Closed" note only if the first session is after cutoff_date
for patient, sessions in patient_data.items():
    session_count = len(sessions)
    first_session_date = datetime.strptime(sessions[0][1], "%Y-%m-%d")  # First session date

    # Determine if this patient is under "Eyes Closed" protocol
    eyes_closed = first_session_date >= cutoff_date

    marked_sessions = []
    for file, session_date in sessions:
        if eyes_closed:
            marked_sessions.append(f'<span class="highlight">{file} ({session_date})</span>')
        else:
            marked_sessions.append(f"{file} ({session_date})")

    session_dates = "<br>".join(marked_sessions)

    # Highlight entire patient row if "Eyes Closed"
    row_style = 'class="highlight"' if eyes_closed else ""

    html_content += f"""
        <tr {row_style}>
            <td>{patient}</td>
            <td>{session_count}</td>
            <td>{session_dates}</td>
        </tr>
    """

html_content += """
    </table>
</body>
</html>
"""


# Save (overwrite) the report
output_file = "eeg_tms_mdd_report.html"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(html_content)

print(f"Report updated successfully: {output_file}")
