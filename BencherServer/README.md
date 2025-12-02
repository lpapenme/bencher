# Bencher Server

## Benchmark Registry

`benchmark-registry.json` tells the server which downstream service should handle a benchmark name. Each entry now accepts an optional `host` in addition to the required `port`, `dimensions`, and `type` fields. When `host` is omitted it defaults to `localhost`, which keeps existing entries valid. Example:

```json
{
  "lasso-dna": {
    "host": "localhost",
    "port": 50053,
    "dimensions": 180,
    "type": "purely_continuous"
  }
}
```

## Dual-Stack Bindings

The server binds to both IPv4 and IPv6 loopback addresses by default. To override, pass one or more `--listen-address` flags, for example:

```bash
uv run start-benchmark-service --listen-address 127.0.0.1 --listen-address ::1
```

Every benchmark service inherits from a dual-stack helper so the downstream gRPC ports are reachable through both stacks as well.
