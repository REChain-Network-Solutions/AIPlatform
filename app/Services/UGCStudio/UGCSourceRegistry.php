<?php

declare(strict_types=1);

namespace App\Services\UGCStudio;

use Illuminate\Support\Collection;

/**
 * Central registry for UGC parent extensions that contribute to the UGC Studio.
 *
 * Bound as a singleton in AppServiceProvider. Parent extensions push themselves
 * in from their own ServiceProvider boot() — Studio reads from the registry only,
 * so missing/uninstalled extensions naturally degrade to "card not shown".
 */
class UGCSourceRegistry
{
    /** @var array<string, UGCSource> */
    private array $sources = [];

    public function register(UGCSource $source): void
    {
        $this->sources[$source->key] = $source;
    }

    public function get(string $key): ?UGCSource
    {
        return $this->sources[$key] ?? null;
    }

    /**
     * @return Collection<string, UGCSource>
     */
    public function all(): Collection
    {
        return collect($this->sources);
    }

    public function has(string $key): bool
    {
        return isset($this->sources[$key]);
    }
}
