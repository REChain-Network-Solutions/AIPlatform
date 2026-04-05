<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class PermissionUsageExtractorService
{
    public function extract(string $modulePath): array
    {
        $keys = [];
        $files = [];
        foreach ((File::glob($modulePath . '/**/*.php') ?: []) as $file) {
            $content = File::get($file);
            preg_match_all("#permission\([\'\"]([^\'\"]+)[\'\"]#", $content, $permissionMatches);
            preg_match_all("#['\"]([a-z0-9_]+_(?:manage|view|create|edit|delete))['\"]#i", $content, $genericMatches);
            foreach (array_merge($permissionMatches[1] ?? [], $genericMatches[1] ?? []) as $key) {
                $key = trim((string) $key);
                if (!$key) {
                    continue;
                }
                $keys[] = $key;
                $files[$key][] = str_replace($modulePath . '/', '', $file);
            }
        }

        $keys = array_values(array_unique($keys));
        ksort($files);

        return [
            'permission_keys' => $keys,
            'permission_files' => $files,
        ];
    }
}
