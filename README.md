# zoom-attendance-checker
zoom attendance checker



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

### 2. Run the Script

Navigate to the directory containing main.py and run it with the following options:

- **-h / --help**: Show help message and exit.
- **-p: (Required)** Path to the CSV file.
- **-t: (Optional)** Minimum required attendance percentage (default is 70%).


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
