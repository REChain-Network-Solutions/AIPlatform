<?php

namespace App\Services\CustomModules\Marketplace;

use App\Models\ModuleRegistryItem;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Schema;
use ZipArchive;

class VersionResolver
{
    public function resolveForZip(string $zipPath): array
    {
        $manifest = $this->readManifest($zipPath);
        if (empty($manifest['module_name'])) {
            return ['status' => 'unknown', 'message' => 'No module manifest was found for version resolution.'];
        }

        $available = $this->findAvailableVersions($manifest['module_name']);
        $current = $manifest['version'] ?? '0.0.0';
        $latest = $available->first();

        return [
            'status' => $latest ? (version_compare($current, $latest, '>=') ? 'current' : 'upgrade_available') : 'unregistered',
            'module_name' => $manifest['module_name'],
            'current_version' => $current,
            'latest_version' => $latest,
            'available_versions' => $available->values()->all(),
            'recommended_action' => $latest && version_compare($current, $latest, '<') ? 'upgrade_to_registry_latest' : 'install_current_package',
            'compatible_with' => $this->findCompatibility($manifest['module_name']),
        ];
    }

    public function findAvailableVersions(string $moduleName): Collection
    {
        if (!class_exists(ModuleRegistryItem::class) || !Schema::hasTable('module_registry_items')) {
            return collect();
        }

        return ModuleRegistryItem::query()
            ->where('module_name', $moduleName)
            ->orWhere('slug', str($moduleName)->slug('-'))
            ->get()
            ->flatMap(fn (ModuleRegistryItem $item) => $item->versions ?: [$item->latest_version])
            ->filter()
            ->unique()
            ->sort(fn ($a, $b) => version_compare((string) $b, (string) $a))
            ->values();
    }

    public function findCompatibility(string $moduleName): array
    {
        if (!class_exists(ModuleRegistryItem::class) || !Schema::hasTable('module_registry_items')) {
            return [];
        }

        return ModuleRegistryItem::query()
            ->where('module_name', $moduleName)
            ->orWhere('slug', str($moduleName)->slug('-'))
            ->get()
            ->pluck('compatible_with')
            ->filter(fn ($row) => is_array($row) && !empty($row))
            ->values()
            ->all();
    }

    protected function readManifest(string $zipPath): array
    {
        $zip = new ZipArchive();
        if ($zip->open($zipPath) !== true) {
            return [];
        }

        try {
            for ($i = 0; $i < $zip->numFiles; $i++) {
                $name = $zip->getNameIndex($i);
                if (!$name || !preg_match('~(^|/)(module|extension)\.json$~i', $name)) {
                    continue;
                }
                $payload = json_decode((string) $zip->getFromIndex($i), true);
                if (!is_array($payload)) {
                    continue;
                }
                return [
                    'module_name' => (string) ($payload['name'] ?? basename(dirname($name))),
                    'version' => (string) ($payload['version'] ?? '0.0.0'),
                ];
            }
        } finally {
            $zip->close();
        }

        return [];
    }
}
