<?php

namespace App\Services\CustomModules\Monitoring;

use Illuminate\Support\Facades\Log;

class InstallationEventDispatcher
{
    protected static function repository(): ?LogEntryRepository
    {
        try {
            return app(LogEntryRepository::class);
        } catch (\Throwable $e) {
            return null;
        }
    }

    protected static function log(string $moduleName, string $eventType, string $level = 'info', array $data = []): void
    {
        try {
            $repo = static::repository();
            if ($repo) {
                $repo->logEvent($moduleName, $eventType, $level, $data);
                return;
            }
        } catch (\Throwable $e) {
            // Fall through to laravel log.
        }

        Log::{$level}("[module-installer] {$eventType}", array_merge(['module_name' => $moduleName], $data));
    }

    public static function securityScanStarted(string $moduleName, string $filePath): void
    {
        static::log($moduleName, 'security_scan_started', 'info', ['file_path' => $filePath]);
    }

    public static function securityScanCompleted(string $moduleName, bool $success, array $results = []): void
    {
        static::log($moduleName, 'security_scan_completed', $success ? 'info' : 'warning', ['success' => $success, 'results' => $results]);
    }

    public static function permissionCheckStarted(string $moduleName, ...$context): void
    {
        static::log($moduleName, 'permission_check_started', 'info', ['context' => $context]);
    }

    public static function permissionCheckCompleted(string $moduleName, bool $success, array $collisions = []): void
    {
        static::log($moduleName, 'permission_check_completed', $success ? 'info' : 'warning', ['success' => $success, 'results' => $collisions]);
    }

    public static function routeCheckStarted(string $moduleName, ...$context): void
    {
        static::log($moduleName, 'route_check_started', 'info', ['context' => $context]);
    }

    public static function routeCheckCompleted(string $moduleName, bool $success, array $collisions = []): void
    {
        static::log($moduleName, 'route_check_completed', $success ? 'info' : 'warning', ['success' => $success, 'results' => $collisions]);
    }

    public static function analysisStarted(string $moduleName, array $context = []): void
    {
        static::log($moduleName, 'analysis_started', 'info', $context);
    }

    public static function analysisCompleted(string $moduleName, bool $success, array $analysis = []): void
    {
        static::log($moduleName, 'analysis_completed', $success ? 'info' : 'warning', ['success' => $success, 'analysis' => $analysis]);
    }

    public static function snapshotCreated(string $moduleName, array $snapshot): void
    {
        static::log($moduleName, 'snapshot_created', 'info', ['snapshot' => $snapshot]);
    }

    public static function filesMovedStarted(string $moduleName, array $context = []): void
    {
        static::log($moduleName, 'files_moved_started', 'info', $context);
    }

    public static function filesMovedCompleted(string $moduleName, bool $success, array $context = []): void
    {
        static::log($moduleName, 'files_moved_completed', $success ? 'info' : 'warning', ['success' => $success, 'context' => $context]);
    }

    public static function repairsStarted(string $moduleName, array $selectedRepairs): void
    {
        static::log($moduleName, 'repairs_started', 'info', ['repairs' => $selectedRepairs]);
    }

    public static function repairsCompleted(string $moduleName, bool|array $successOrResults, array $results = []): void
    {
        $success = is_array($successOrResults) ? empty($successOrResults['failed'] ?? []) : (bool) $successOrResults;
        $payload = is_array($successOrResults) && empty($results) ? $successOrResults : $results;
        static::log($moduleName, 'repairs_completed', $success ? 'info' : 'warning', ['success' => $success, 'results' => $payload]);
    }

    public static function installationCompleted(string $moduleName, int|array|null $installId = null, bool $success = true): void
    {
        $payload = is_array($installId)
            ? $installId
            : ['install_id' => $installId, 'success' => $success];
        static::log($moduleName, 'installation_completed', ($payload['success'] ?? $success) ? 'info' : 'warning', $payload);
    }

    public static function installationFailed(string $moduleName, string|array $reason): void
    {
        $payload = is_array($reason) ? $reason : ['reason' => $reason];
        static::log($moduleName, 'installation_failed', 'error', $payload);
    }

    public static function rollbackStarted(string $moduleName, int|array $installId): void
    {
        $payload = is_array($installId) ? $installId : ['install_id' => $installId];
        static::log($moduleName, 'rollback_started', 'warning', $payload);
    }

    public static function rollbackCompleted(string $moduleName, bool|array $successOrDetails, array $details = []): void
    {
        $success = is_array($successOrDetails) ? true : (bool) $successOrDetails;
        $payload = is_array($successOrDetails) && empty($details) ? $successOrDetails : $details;
        static::log($moduleName, 'rollback_completed', $success ? 'warning' : 'error', ['success' => $success, 'details' => $payload]);
    }

    public static function rollbackFailed(string $moduleName, string|array $reason): void
    {
        $payload = is_array($reason) ? $reason : ['reason' => $reason];
        static::log($moduleName, 'rollback_failed', 'error', $payload);
    }
}
