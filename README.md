# Calvin Woehrle's Brain


## Convert from dcm to Niix

```shell
dcm2niix -o niix -f "%d_%s" /Users/calvinwoehrle/Documents/MRI/CRM15_2026_06_23
```


## Convert with compression
```shell
dcm2niix -z y -o niix -f "%d_%s" /Users/calvinwoehrle/Documents/MRI/CRM15_2026_06_23
```
