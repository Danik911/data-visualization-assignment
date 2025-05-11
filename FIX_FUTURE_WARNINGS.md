# Fixing FutureWarnings in Dashboard Callbacks

The deployed dashboard is showing multiple FutureWarnings related to JSON handling:

```
Passing literal json to 'read_json' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object.
```

## How to Fix

To fix these warnings, you need to update the `callbacks.py` file by changing how JSON is handled. Follow these steps:

1. Add these imports at the top of the file:
   ```python
   from io import StringIO  # Add this import
   ```

2. Replace all instances of direct JSON parsing:
   
   From:
   ```python
   pd.read_json(json_string)
   ```
   
   To:
   ```python
   pd.read_json(StringIO(json_string))
   ```

3. Apply this change to all callbacks that use `pd.read_json`, including:
   - Line 187
   - Line 247
   - Line 362
   - Line 395
   - Line 417
   - Line 437
   - Line 494
   - Line 550

## Alternative Solution Using Helper Function

You can also create a helper function to centralize this fix:

```python
def safe_read_json(json_str):
    """Safely read JSON string into DataFrame using StringIO to avoid FutureWarning."""
    return pd.read_json(StringIO(json_str))
```

Then replace all instances of `pd.read_json(json_string)` with `safe_read_json(json_string)`.

## After Fixing

After applying these changes, redeploy your application on Render. The warnings should be gone, and it will make your code future-proof when this deprecation becomes an error in future pandas versions. 