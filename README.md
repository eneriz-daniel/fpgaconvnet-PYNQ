# fpgaConvNet PYNQ driver

This repo contains the [fpgaConvNet PYNQ driver](fpgaconvnet_pynq_driver.py), which allows to run [fpgaConvNet](https://alexmontgomerie.github.io/fpgaconvnet-website/) IPs under the [PYNQ framework](http://www.pynq.io/).

Also, there is a [notebook](PYNQ-deployment-tutorial.ipynb) that shows how to use the driver. In summary, you can just:
```python
>>> from pynq import Overlay
>>> from fpgaconvnet_pynq_driver import *
>>> overlay = Overlay('bitstream.bit')
>>> fpgaconvnet_ip = overlay.fpgaconvnet_ip_0
>>> fpgaconvnet_ip.load_partition('partitions.json', 0)
>>> Y = fpgaconvnet_ip.run(X)
```

The rest of files in this repo are used as assets for the tutorial notebook.