You know that bytetrack's source code is a hassle to install, but you can actually rewrite parts of it to achieve exactly the same functionality. This project rewrites complex dependencies in numpy and scipy to achieve consistent functionality 

So I have this project.Compared to the original code https://github.com/ifzhang/ByteTrack, I canceled the training module, and also simplified the process of demo using. For those who just want to use the ByteTrack feature, All you need is a python packages are:

```
ultralytics 
cv2
numpy 
scipy
```

The main entrance is at `main.ipynb`, everyone is free to modify the file to achieve what they want. 


Finally, you can find sample videos in the file `video/1_t.mp4`
