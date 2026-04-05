<?php

namespace App\Services\CustomModules\Marketplace;

use App\Models\ModuleRegistryFeed;
use App\Models\ModuleRegistryItem;
use Illuminate\Support\Facades\Schema;

class RemoteRegistryIngestor
{
    public function __construct(protected RegistryFeedAdapter $adapter)
    {
    }

    public function ingestActiveFeeds(): array
    {
        if (!class_exists(ModuleRegistryFeed::class) || !Schema::hasTable('module_registry_feeds')) {
            return ['feeds' => 0, 'items' => 0, 'skipped' => true, 'reason' => 'registry_tables_missing'];
        }

        $feeds = ModuleRegistryFeed::query()->where('is_active', true)->orderByDesc('priority')->get();
        $items = 0;
        $errors = [];

        foreach ($feeds as $feed) {
            try {
                $payload = $this->adapter->fetch($feed);
                $items += $this->persistPayload($feed, $payload['items'] ?? []);
                $feed->forceFill([
                    'last_synced_at' => now(),
                    'last_error' => null,
                    'sync_meta' => $payload['meta'] ?? [],
                ])->save();
            } catch (\Throwable $e) {
                $errors[] = ['feed' => $feed->slug, 'message' => $e->getMessage()];
                $feed->forceFill([
                    'last_synced_at' => now(),
                    'last_error' => $e->getMessage(),
                ])->save();
            }
        }

        return ['feeds' => $feeds->count(), 'items' => $items, 'errors' => $errors];
    }

    public function persistPayload(ModuleRegistryFeed $feed, array $items): int
    {
        if (!Schema::hasTable('module_registry_items')) {
            return 0;
        }

        $count = 0;
        foreach ($items as $item) {
            if (!is_array($item) || empty($item['module_name'])) {
                continue;
            }

            ModuleRegistryItem::updateOrCreate(
                ['feed_id' => $feed->id, 'slug' => $item['slug'] ?? str($item['module_name'])->slug('-')],
                [
                    'module_name' => $item['module_name'],
                    'latest_version' => $item['latest_version'] ?? '0.0.0',
                    'versions' => $item['versions'] ?? [],
                    'compatible_with' => $item['compatible_with'] ?? [],
                    'dependencies' => $item['dependencies'] ?? [],
                    'distribution' => $item['distribution'] ?? [],
                    'signature' => $item['signature'] ?? [],
                    'status' => $item['status'] ?? 'active',
                    'last_seen_at' => now(),
                    'meta' => $item['meta'] ?? [],
                ]
            );
            $count++;
        }

        return $count;
    }
}
