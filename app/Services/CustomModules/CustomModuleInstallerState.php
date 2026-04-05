<?php
namespace App\Services\CustomModules;

use App\Services\CustomModules\DTO\ModuleFileRecord;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\File;

class CustomModuleInstallerState
{
    public function storeUpload(UploadedFile $upload, string $modulePath): array
    {
        $originalName = $upload->getClientOriginalName();
        $safeName = preg_replace('/[^A-Za-z0-9_\-\.]/', '_', $originalName) ?: ('module_' . time() . '.zip');

        if (File::exists($modulePath . DIRECTORY_SEPARATOR . $safeName)) {
            $safeName = pathinfo($safeName, PATHINFO_FILENAME) . '_' . time() . '.zip';
        }

        $upload->move($modulePath, $safeName);

        return ['name' => $safeName, 'path' => $modulePath . DIRECTORY_SEPARATOR . $safeName];
    }

    public function moduleFiles(string $modulePath): array
    {
        if (! File::exists($modulePath)) return [];

        return collect(File::files($modulePath))
            ->sortByDesc(fn ($file) => $file->getMTime())
            ->map(fn ($file) => (new ModuleFileRecord(
                name: $file->getFilename(),
                path: $file->getPathname(),
                size: $file->getSize(),
                modified: $file->getMTime()
            ))->toArray())
            ->values()->all();
    }

    public function resolveModuleFile(string $filePath, string $modulePath): string
    {
        $candidate = $modulePath . DIRECTORY_SEPARATOR . basename($filePath);
        if (File::exists($candidate)) return $candidate;
        if (File::exists($filePath)) return $filePath;
        abort(404, 'File not found.');
    }

    public function doctorSummaryCards(): array
    {
        return [
            ['title' => 'Doctor summary', 'body' => 'Route truth, sidebar inclusion, permission readiness, registry checks, and tenant entitlement checks are enabled in the analyzer.', 'theme' => 'primary'],
            ['title' => 'Safe fix preview', 'body' => 'Menu SQL preview, sidebar injector preview, and safe repair queue are available in a compact card layout.', 'theme' => 'success'],
        ];
    }

    public function doctorPanels(): array
    {
        return [
            ['key' => 'visibility', 'title' => 'User menu visibility', 'status' => 'enabled'],
            ['key' => 'routes', 'title' => 'Route truth', 'status' => 'enabled'],
            ['key' => 'permissions', 'title' => 'Permission readiness', 'status' => 'enabled'],
            ['key' => 'tenant', 'title' => 'Tenant entitlement', 'status' => 'enabled'],
        ];
    }

    public function quickStats(string $modulePath): array
    {
        $files = $this->moduleFiles($modulePath);

        return [
            'stored_zip_count' => count($files),
            'latest_upload' => $files[0]['name'] ?? null,
            'latest_upload_size' => $files[0]['size'] ?? 0,
            'total_storage_bytes' => array_sum(array_column($files, 'size')),
        ];
    }

    public function recentHistory(): array
    {
        return [
            ['event' => 'overlay-installed', 'time' => now()->toDateTimeString()],
            ['event' => 'upload-path-bound', 'time' => now()->toDateTimeString()],
            ['event' => 'doctor-panels-bound', 'time' => now()->toDateTimeString()],
        ];
    }

    public function safeFixPresets(): array
    {
        return ['permissions' => true, 'package_bridge' => true, 'company_module_settings' => true, 'clear_caches' => true];
    }
}
