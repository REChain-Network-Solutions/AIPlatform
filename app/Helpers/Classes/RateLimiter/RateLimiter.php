<?php

namespace App\Helpers\Classes\RateLimiter;

use App\Helpers\Classes\Helper;
use Illuminate\Support\Facades\Cache;

class RateLimiter
{
    protected string $prefix;

    protected int $maxAttempts = 4;

    protected int $decayMinutes = 1400;

    public function __construct(
        string $prefix,
        int $maxAttempts = 4,
        int $decayMinutes = 1440
    ) {
        $this->prefix = $prefix;
        $this->maxAttempts = $maxAttempts;
        $this->decayMinutes = $decayMinutes;
    }

    public function attempt(?string $identifier = null): bool
    {
        $identifier = $identifier ?? Helper::getRequestIp();

        $cacheKey = $this->buildCacheKey($identifier);

        $requestCount = Cache::get($cacheKey, 0);

        $this->incrementAttempts($cacheKey, $requestCount);

        return $requestCount < $this->maxAttempts;
    }

    public function remainingAttempts(string $identifier): int
    {
        $cacheKey = $this->buildCacheKey($identifier);
        $requestCount = Cache::get($cacheKey, 0);

        return max(0, $this->maxAttempts - $requestCount);
    }

    public function attempts(string $identifier): int
    {
        $cacheKey = $this->buildCacheKey($identifier);

        return Cache::get($cacheKey, 0);
    }

    public function reset(string $identifier): void
    {
        $cacheKey = $this->buildCacheKey($identifier);
        Cache::forget($cacheKey);
    }

    protected function buildCacheKey(string $identifier): string
    {
        $date = now()->format('Y-m-d');

        return "{$this->prefix}:{$identifier}:{$date}";
    }

    protected function incrementAttempts(string $cacheKey, int $currentCount): void
    {
        $expiresAt = now()->addMinutes($this->decayMinutes)->startOfDay();

        Cache::put($cacheKey, $currentCount + 1, $expiresAt);
    }

    public static function make(string $prefix, int $maxAttempts = 4, int $decayMinutes = 1440): self
    {
        return new self($prefix, $maxAttempts, $decayMinutes);
    }
}
