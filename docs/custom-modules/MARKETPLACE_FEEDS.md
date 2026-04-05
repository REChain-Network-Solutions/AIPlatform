# Marketplace Feed Adapter

This pass adds a registry layer for remote or local marketplace feeds.

## Supported source types

- `file` — read a JSON registry file from disk
- `json` — store a full JSON payload directly on the feed row
- `http`/`https` — fetch a remote registry endpoint with optional bearer token

## Registry payload shape

```json
{
  "items": [
    {
      "module_name": "TitanCustomers",
      "slug": "titan-customers",
      "latest_version": "1.2.0",
      "versions": ["1.0.0", "1.1.0", "1.2.0"],
      "compatible_with": {
        "laravel": "^10.0",
        "worksuite": ">=5.5.18"
      },
      "dependencies": {
        "Core": "^1.0.0"
      },
      "distribution": {
        "channel": "stable",
        "download_url": "https://example.test/module.zip"
      },
      "signature": {
        "sha256": "..."
      }
    }
  ]
}
```

## Runtime use

- Dashboard summary shows feed and item counts
- Analyze/install can resolve current ZIP version against registry items
- A newer registry version triggers a warning in diagnostics
