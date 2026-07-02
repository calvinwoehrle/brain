# Calvin Woehrle's Brain


## Convert from dcm to Niix

```shell
dcm2niix -o niix -f "%d_%s" <PATH_TO_DICOM_FOLDER>
```


## Convert with compression
```shell
dcm2niix -z y -o niix -f "%d_%s" <PATH_TO_DICOM_FOLDER>
```
