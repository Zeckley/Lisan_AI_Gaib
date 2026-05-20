---
tags:
  - BigTask
---
We are going to overhaul how the directives work so we can standardize some logic on the AI side and help set up a scoring system for the AI training.

Currently, we have directives that can target certain buildings with different tax rates and urgencies. I need to bolster this to provide more flexibility to the defualt logic setup

Default Logic loop
Check food rates
if food rate negative, attempt to upgrade farms
if no farms can be safely upgraded and staffed properly, build a new farm