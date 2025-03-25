# Cell Counter

[![CI](https://img.shields.io/github/actions/workflow/status/YourGitHubUsername/YourRepositoryName/ci.yaml?branch=main&style=flat-square)](https://github.com/YourGitHubUsername/YourRepositoryName/actions)

A Python-based tool for **counting cells** from an image. It uses:

1. **Automatic color-based detection** (in the HSV color space) to identify cells based on a target color and tolerance.  
2. **Manual adjustments** where you can left-click to add cells or right-click to remove cells.  
3. **Tkinter** for file selection dialogs and a simple **GUI** window showing the final count.  
4. **OpenCV** for image processing and display of detected contours.  
5. An **Average Cell** preview, displaying a composite of automatically detected cells.

---

## Features

- **Automatic Cell Detection** based on color and contour size filtering.  
- **Manual Edits**: left-click to add a cell, right-click to remove a nearby cell.  
- **Live Cell Counter** displayed in a separate mini-window.  
- **Average Cell Preview** window showing the composite of detected cells.  
- **Easy Configuration** via a `target_color_bgr` and an HSV `tolerance` array.  
- **Docker Support**: Run in a container with minimal fuss.  
- **GitHub Actions CI** workflow to test, build, and optionally tag a release.

---

## Table of Contents

- [Installation (Local)](#installation-local)
- [Usage (Local)](#usage-local)
- [Docker Usage](#docker-usage)
- [Examples](#examples)
- [Development & Testing](#development--testing)
- [License](#license)

---

## Installation (Local)

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/YourRepositoryName.git
   cd YourRepositoryName
   ```

2. **Install System Dependencies**:

   If you are using Ubuntu or a Debian-based system, run:
   ```bash
   sudo apt-get update
   # Installs everything listed in apt-requirements.txt
   xargs -a apt-requirements.txt sudo apt-get install -y
   ```

3. **Install Python Packages**:

   ```bash
   pip install -r pip-requirements.txt
   ```
   Make sure you have a suitable Python version installed (e.g., 3.9+).

---

## Usage (Local)

1. **Run the script**:
   ```bash
   python Cell-Counter.py
   ```

2. **Select an image** when prompted by the file dialog (supports `.tiff`, `.tif`, `.jpg`, `.png`, etc.).

3. **Interact with the OpenCV window**:
   - **Left-click** on the image to add a cell at that location.
   - **Right-click** on or near an existing cell to remove it.
   - Press the **ESC** key to close.

4. **Check the live count** in the **"Counter"** window.  
5. The **"Average Cell"** window displays the composite of automatically detected cells (if any).  

When you exit, the **final count** will be printed in your terminal.

---

## Docker Usage

### 1. Build the Docker Image

From the root of the repository (where the Dockerfile resides), run:
```bash
docker build -t cell-counter .
```

### 2. Run the Container

If you want to **see** the Tkinter and OpenCV GUIs on your host (e.g., Linux with X11), you can do:
```bash
docker run --rm -it \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    cell-counter
```
- `-e DISPLAY` and the `-v /tmp/.X11-unix:/tmp/.X11-unix` volume mount forward your **X11** display to the container so that GUI windows appear on your host.

Adjust for your OS or environment if you have a different graphical setup. For example, on macOS you might need an XQuartz-based solution; on Windows, you might need an alternative X server.

Once the container starts, you’ll see the same **file selection dialog** and **GUI** windows as if running locally.

---

## Examples

### Automatic Detection Preview

*If you have a screenshot of the app automatically detecting cells, place it here.*

![Automatic detection screenshot](docs/automatic_detection.png)

### Manual Adjustments

- **Left-Click** to add a cell  
- **Right-Click** to remove a cell (if close to its center)

![Manual adjustments screenshot](docs/manual_edit.png)

### Average Cell

If enough contours are detected, an “Average Cell” window will appear:

![Average cell screenshot](docs/average_cell.png)

---

## Development & Testing

1. **Run Tests**  
   This project uses Python’s built-in `unittest` (or `pytest`, adapt if you prefer). If you have a `tests/` folder, you can do:
   ```bash
   python -m unittest discover
   ```
   
2. **Continuous Integration**  
   A [GitHub Actions workflow](.github/workflows/ci.yaml) is provided that:  
   - Installs system + Python dependencies  
   - Runs tests  
   - Builds the Docker image  
   - Creates a GitHub Release if manually triggered with a version number  

You can customize this workflow to automatically push the Docker image to Docker Hub or another registry.

---

## License

This project is licensed under the [MIT License](LICENSE) – feel free to modify or re-license as you see fit. If you fork or derive from it, a link back here is always appreciated!

---

### Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to open issues or submit pull requests.

---

**Enjoy counting cells!** If you find this project helpful, consider giving it a ⭐ on GitHub.
