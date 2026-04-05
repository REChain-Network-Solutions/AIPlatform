<?php

namespace App\Services\CustomModules\Monitoring;

use App\Models\ModuleInstallationLog;
use Carbon\Carbon;
use Illuminate\Pagination\LengthAwarePaginator;
use Illuminate\Support\Facades\File;

class LogEntryRepository
{
    public function logEvent(
        string $moduleName,
        string $eventType,
        string $level = 'info',
        array $data = [],
        ?int $userId = null,
        ?int $installId = null
    ): ModuleInstallationLog {
        return ModuleInstallationLog::create([
            'module_name' => $moduleName,
            'event_type' => $eventType,
            'level' => $this->normaliseLevel($level),
            'data' => $data,
            'user_id' => $userId ?? auth()->id(),
            'logged_at' => now(),
            'install_id' => $installId,
        ]);
    }

    public function logError(string $moduleName, string $message, array $context = []): ModuleInstallationLog
    {
        return $this->logEvent($moduleName, 'error', 'error', ['message' => $message, 'context' => $context]);
    }

    public function logWarning(string $moduleName, string $message, array $context = []): ModuleInstallationLog
    {
        return $this->logEvent($moduleName, 'warning', 'warning', ['message' => $message, 'context' => $context]);
    }

    public function logInfo(string $moduleName, string $message, array $context = []): ModuleInstallationLog
    {
        return $this->logEvent($moduleName, 'info', 'info', ['message' => $message, 'context' => $context]);
    }

    public function getLogsForModule(string $moduleName, int $perPage = 100): LengthAwarePaginator
    {
        return ModuleInstallationLog::where('module_name', $moduleName)
            ->orderByDesc('logged_at')
            ->paginate($perPage);
    }

    public function getLogsForInstallation(int $installId, int $perPage = 100): LengthAwarePaginator
    {
        return ModuleInstallationLog::where('install_id', $installId)
            ->orderByDesc('logged_at')
            ->paginate($perPage);
    }

    public function getRecentErrors(int $minutes = 60, int $perPage = 50): LengthAwarePaginator
    {
        return ModuleInstallationLog::where('level', 'error')
            ->where('logged_at', '>=', now()->subMinutes($minutes))
            ->orderByDesc('logged_at')
            ->paginate($perPage);
    }

    public function getAuditTrail(string $moduleName, int $days = 30): array
    {
        return ModuleInstallationLog::where('module_name', $moduleName)
            ->where('logged_at', '>=', now()->subDays($days))
            ->orderBy('logged_at')
            ->get()
            ->map(function (ModuleInstallationLog $log) {
                return [
                    'timestamp' => optional($log->logged_at)->toDateTimeString(),
                    'event' => $log->event_type,
                    'level' => $log->level,
                    'message' => data_get($log->data, 'message'),
                    'user' => $log->user_id ? optional(\App\Models\User::find($log->user_id))->name : 'system',
                ];
            })->all();
    }

    public function search(array $filters = [], int $perPage = 50): LengthAwarePaginator
    {
        $query = ModuleInstallationLog::query();

        foreach (['module_name', 'event_type', 'level', 'user_id', 'install_id'] as $field) {
            if (!empty($filters[$field])) {
                $query->where($field, $filters[$field]);
            }
        }

        if (!empty($filters['start_date'])) {
            $query->where('logged_at', '>=', Carbon::parse($filters['start_date'])->startOfDay());
        }

        if (!empty($filters['end_date'])) {
            $query->where('logged_at', '<=', Carbon::parse($filters['end_date'])->endOfDay());
        }

        return $query->orderByDesc('logged_at')->paginate($perPage);
    }

    public function exportToCSV(?string $moduleName, string $filePath): bool
    {
        $query = ModuleInstallationLog::query()->orderByDesc('logged_at');
        if (!empty($moduleName)) {
            $query->where('module_name', $moduleName);
        }

        File::ensureDirectoryExists(dirname($filePath));
        $fp = fopen($filePath, 'w');
        if (!$fp) {
            return false;
        }

        fputcsv($fp, ['Timestamp', 'Module', 'Event', 'Level', 'Message', 'User']);
        foreach ($query->cursor() as $log) {
            fputcsv($fp, [
                optional($log->logged_at)->toDateTimeString(),
                $log->module_name,
                $log->event_type,
                $log->level,
                data_get($log->data, 'message', ''),
                $log->user_id ? optional(\App\Models\User::find($log->user_id))->name : 'system',
            ]);
        }

        fclose($fp);
        return true;
    }

    public function cleanupOldLogs(int $daysToKeep = 90): int
    {
        return ModuleInstallationLog::where('logged_at', '<', now()->subDays($daysToKeep))->delete();
    }

    public function getStatistics(int $days = 7): array
    {
        $logs = ModuleInstallationLog::where('logged_at', '>=', now()->subDays($days))->get();

        return [
            'total_logs' => $logs->count(),
            'by_level' => $logs->groupBy('level')->map->count()->toArray(),
            'by_event_type' => $logs->groupBy('event_type')->map->count()->toArray(),
            'by_module' => $logs->groupBy('module_name')->map->count()->toArray(),
            'errors_count' => $logs->where('level', 'error')->count(),
            'warnings_count' => $logs->where('level', 'warning')->count(),
        ];
    }

    protected function normaliseLevel(string $level): string
    {
        return in_array($level, ['debug', 'info', 'warning', 'error'], true) ? $level : 'info';
    }
}
