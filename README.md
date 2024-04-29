# zoom-attendance-checker

<br>
<br>

### Please note!

The attendee report of a Zoom webinar is saved as a CSV file, but it may not open correctly due to inconsistent data formats within. 

For this issue, just opening the "attendance.csv" file in MS Excel and then saving it without any additional actions will allow Excel to automatically convert and save the data in the appropriate format, thus resolving the problem.

It is recommended to preprocess the file using the above method before using the program.

And, The format for names is a number followed by a name (e.g., 2024012345 GildongHong).

<br>



## How to Use
To use the Zoom Attendance Checker, follow these steps:

### 1. Clone the Repository

Use one of the following commands to clone the repository:

```sh
git clone https://github.com/kdgyun/zoom-attendance-checker.git
```
or

```sh
git clone git@github.com:kdgyun/zoom-attendance-checker.git
```

<br>

### 2. Navigate to the cloned repository directory and install the required Python packages:

```sg
cd zoom-attendance-checker
pip install -r requirements.txt
```

<br>

### 3. Run the Script

Navigate to the directory containing main.py and run it with the following options:

- **-h / --help**: Show help message and exit.
- **-p / --path: (Required)** Path to the CSV file.
- **-t / --threshold: (Optional)** Minimum required attendance percentage (default is 70%).
- **-l / --length: (Optional)** Length of the unique number (e.g., student ID number). If not specified, a default length is assumed 10.


#### Examples:
**Basic usage:**
```sh
python main.py -p path-to-attendance.csv
```
</br>

**To consider attendees with more than 80% attendance:**
```sh
python main.py -p path-to-attendance.csv -t 80
```

<br>

The script generates an **Excel (.xlsx) file** with the attendance results, including a list of attendees marked present and a section for those needing manual verification.

Contributing
We welcome contributions! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.
