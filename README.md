# treecounter-ML
A simple application that detects and counts the number of trees in a vineyard. Makes use of [darknet](https://pjreddie.com/darknet/) and [OpenCV](https://opencv.org/).

<p align="center">
    <img src="imgs/test.gif" width="600" height="400">                           
</p>

---

## Setup
- Clone the repository onto your local machine : `git clone https://github.com/sampreets3/treecounter-ML`
- Create the **videos** and **models** directories :
  ```
  $ mkdir models
  $ mkdir videos
  ```
- Add the video in the `videos` directory.
- Add the model-specific data in the `models` directory
- Run the detector :
```sh
python3 test.py -iv imgs/input-video.mp4 -ov result.mp4
```
---

### Common Problems:

 - [![GtkMessage-fixed](https://img.shields.io/badge/GtkMessage-fixed-green.svg)](https://shields.io/) **Gtk-Message: Failed to load module "canberra-gtk-module" :** You can fix the issue by installing the gtk and gtk3 modules `sudo apt install libcanberra-gtk-module libcanberra-gtk3-module`

### Maintainer Information

Sampreet Sarkar `sampreets3@gmail.com`
