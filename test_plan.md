# Test management for the project

Given the nature of the project (most classes functions are being run as one-offs in data processing pipelines) the typical unit tests are replaced by validation notebooks.

Each class or group of methods has an associated test script.

Validation notebooks trigger those test scripts and print the typically visual outcomes for inspection.

These can then be reviewed for consistency with goals and ensuring that consecutive steps develop as intended.