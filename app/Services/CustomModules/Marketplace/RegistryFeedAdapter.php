<?php

namespace App\Services\CustomModules\Marketplace;

use App\Models\ModuleRegistryFeed;
use Illuminate\Support\Arr;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Http;

class RegistryFeedAdapter
{
    public function fetch(ModuleRegistryFeed $feed): array
    {
        $sourceType = strtolower((string) $feed->source_type);
        $sourceUrl = (string) $feed->source_url;

        if ($sourceType === 'file') {
            return $this->parsePayload(File::exists($sourceUrl) ? File::get($sourceUrl) : '[]', $feed, ['source' => 'file']);
        }

        if ($sourceType === 'json') {
            return $this->parsePayload($sourceUrl, $feed, ['source' => 'inline_json']);
        }

        $request = Http::timeout((int) config('custom-modules.registry.http_timeout', 8));
        $token = Arr::get($feed->auth_config ?? [], 'token');
        if ($token) {
            $request = $request->withToken($token);
        }

        $response = $request->get($sourceUrl);
        $response->throw();

        return $this->parsePayload($response->body(), $feed, [
            'source' => 'http',
            'status' => $response->status(),
            'etag' => $response->header('ETag'),
        ]);
    }

    protected function parsePayload(string $payload, ModuleRegistryFeed $feed, array $meta = []): array
    {
        $decoded = json_decode($payload, true);
        if (!is_array($decoded)) {
            return ['feed' => $feed->slug, 'items' => [], 'meta' => $meta + ['error' => 'invalid_json_payload']];
        }

        $items = $decoded['items'] ?? $decoded['modules'] ?? $decoded;
        if (!is_array($items)) {
            $items = [];
        }

        return [
            'feed' => $feed->slug,
            'items' => array_values(array_filter(array_map(fn ($item) => $this->normaliseItem($item), $items))),
            'meta' => $meta + ['fetched_at' => now()->toIso8601String()],
        ];
    }

    protected function normaliseItem($item): ?array
    {
        if (!is_array($item)) {
            return null;
        }

        $name = (string) ($item['module_name'] ?? $item['name'] ?? $item['slug'] ?? '');
        if ($name === '') {
            return null;
        }

        $versions = $item['versions'] ?? [];
        if (is_string($versions)) {
            $versions = [$versions];
        }

        $latest = (string) ($item['latest_version'] ?? $item['version'] ?? (is_array($versions) ? end($versions) : '0.0.0'));

        return [
            'module_name' => $name,
            'slug' => (string) ($item['slug'] ?? str($name)->slug('-')),
            'latest_version' => $latest,
            'versions' => array_values(array_unique(array_filter(array_merge([$latest], is_array($versions) ? $versions : [])))),
            'compatible_with' => is_array($item['compatible_with'] ?? null) ? $item['compatible_with'] : [],
            'dependencies' => is_array($item['dependencies'] ?? null) ? $item['dependencies'] : [],
            'distribution' => is_array($item['distribution'] ?? null) ? $item['distribution'] : [],
            'signature' => is_array($item['signature'] ?? null) ? $item['signature'] : [],
            'status' => (string) ($item['status'] ?? 'active'),
            'meta' => is_array($item['meta'] ?? null) ? $item['meta'] : [],
        ];
    }
}
