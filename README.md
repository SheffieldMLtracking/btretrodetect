# retrodetect
- This new version (completely rewritten) assumes all the photos are flash photos.

# install
```pip install git+https://github.com/lionfish0/btretrodetect.git```

# usage
One passes a path and the tool will recursively search through the subdirectories, finding all the images, sorting them (within that folder) and applying the retrodetect algorithm.

Example:
```btretrodetect ~/Documents/Research/rsync_bee/test/beephotos/2023-06-29/sessionA/setA/cam5/02D49670796/ --after 10:32:29 --before 10:33:29 --threshold -10```

```usage: btretrodetect [-h] [--after AFTER] [--before BEFORE] [--refreshcache] [--threshold THRESHOLD] [--sourcename SOURCENAME] imgpath

Runs the retoreflector detection algorithm

positional arguments:
  imgpath               Path to images (it will recursively search for images in these paths)

options:
  -h, --help            show this help message and exit
  --after AFTER         Only process images that were created after this time HH:MM:SS
  --before BEFORE       Only process images that were created before this time HH:MM:SS
  --refreshcache        Whether to refresh the cache
  --threshold THRESHOLD
                        Threshold of score before adding to data
  --sourcename SOURCENAME
                        The name to give this source of labels (default:retrodetect)
```


# Output
As it runs it outputs a list of all the files it is processing. After each file it records a `.` or a `x` for each bright point it's considered. Ones that reach the threshold (the `x`s) will be added to the json label file.

Example:
```2023-06-29/sessionA/setA/cam5/02D49670796/photo_object_02D49670796_20230629_10:32:31.586358__0014.np .....
2023-06-29/sessionA/setA/cam5/02D49670796/photo_object_02D49670796_20230629_10:32:31.841386__0015.np x....
2023-06-29/sessionA/setA/cam5/02D49670796/photo_object_02D49670796_20230629_10:32:32.136017__0016.np .....
2023-06-29/sessionA/setA/cam5/02D49670796/photo_object_02D49670796_20230629_10:32:32.457968__0017.np x....
2023-06-29/sessionA/setA/cam5/02D49670796/photo_object_02D49670796_20230629_10:32:32.742483__0018.np x....
```
