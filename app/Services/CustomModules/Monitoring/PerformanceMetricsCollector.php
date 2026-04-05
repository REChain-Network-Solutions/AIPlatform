<?php

namespace App\Services\CustomModules\Monitoring;

use Illuminate\Support\Facades\DB;
use App\Models\ModuleInstallLog;
use Carbon\Carbon;

class PerformanceMetricsCollector
{
    protected array $metrics = [];
    protected Carbon $startTime;
    protected array $phases = [];

    public function __construct()
    {
        $this->startTime = now();
    }

    /**
     * Start measuring a phase
     */
    public function startPhase(string $phaseName): void
    {
        $this->phases[$phaseName] = [
            'started_at' => microtime(true),
            'memory_start' => memory_get_usage(true),
        ];
    }

    /**
     * End measuring a phase
     */
    public function endPhase(string $phaseName): array
    {
        if (!isset($this->phases[$phaseName])) {
            return [];
        }

        $phase = $this->phases[$phaseName];
        $duration = (microtime(true) - $phase['started_at']) * 1000; // ms
        $memoryUsed = (memory_get_usage(true) - $phase['memory_start']) / 1024 / 1024; // MB

        $this->metrics[$phaseName] = [
            'duration_ms' => round($duration, 2),
            'memory_mb' => round($memoryUsed, 2),
            'started_at' => date('Y-m-d H:i:s', intval($phase['started_at'])),
        ];

        return $this->metrics[$phaseName];
    }

    /**
     * Get metrics for a specific module installation
     */
    public function getInstallationMetrics(int $installId): array
    {
        $install = ModuleInstallLog::find($installId);

        if (!$install) {
            return [];
        }

        $repairResults = $install->repair_results ?? [];
        $validationResults = $install->validation_results ?? [];

        return [
            'module_name' => $install->module_name,
            'status' => $install->status,
            'installed_at' => $install->installed_at,
            'validation_score' => $install->validation_score,
            'phases' => $this->extractPhaseMetrics($repairResults),
            'validation' => $this->extractValidationMetrics($validationResults),
            'repairs' => $this->extractRepairMetrics($repairResults),
            'total_duration_seconds' => $install->installed_at ? now()->diffInSeconds($install->installed_at) : 0,
        ];
    }

    /**
     * Get all metrics for current collection
     */
    public function getMetrics(): array
    {
        return $this->metrics;
    }

    /**
     * Get average installation time (last N installations)
     */
    public function getAverageInstallationTime(int $limit = 10): float
    {
        $installs = ModuleInstallLog::where('status', 'installed')
            ->orderBy('installed_at', 'desc')
            ->limit($limit)
            ->get();

        if ($installs->isEmpty()) {
            return 0;
        }

        $totalTime = $installs->sum(function ($install) {
            return $install->installed_at ? now()->diffInSeconds($install->installed_at) : 0;
        });

        return round($totalTime / $installs->count(), 2);
    }

    /**
     * Detect slow operations
     */
    public function detectSlowOperations(float $thresholdMs = 1000): array
    {
        return array_filter($this->metrics, function ($metric) use ($thresholdMs) {
            return ($metric['duration_ms'] ?? 0) > $thresholdMs;
        });
    }

    /**
     * Get bottleneck analysis
     */
    public function getBottlenecks(): array
    {
        $sorted = collect($this->metrics)->sortByDesc('duration_ms');

        return [
            'slowest_phase' => $sorted->first(),
            'top_3_slowest' => $sorted->take(3)->toArray(),
            'average_duration_ms' => round(collect($this->metrics)->avg('duration_ms'), 2),
        ];
    }

    /**
     * Extract phase metrics from repair results
     */
    protected function extractPhaseMetrics(array $repairResults): array
    {
        $phases = [];

        foreach ($repairResults as $repairName => $result) {
            if (isset($result['duration_ms'])) {
                $phases[$repairName] = $result['duration_ms'];
            }
        }

        return $phases;
    }

    /**
     * Extract validation metrics
     */
    protected function extractValidationMetrics(array $validationResults): array
    {
        $metrics = [];

        foreach ($validationResults as $validatorName => $result) {
            $metrics[$validatorName] = [
                'valid' => $result['valid'] ?? false,
                'issues_count' => count($result['issues'] ?? []),
            ];
        }

        return $metrics;
    }

    /**
     * Extract repair metrics
     */
    protected function extractRepairMetrics(array $repairResults): array
    {
        $metrics = [];

        foreach ($repairResults as $repairName => $result) {
            $metrics[$repairName] = [
                'success' => $result['success'] ?? false,
                'duration_ms' => $result['duration_ms'] ?? 0,
                'operations' => count($result['results'] ?? []),
            ];
        }

        return $metrics;
    }

    /**
     * Generate performance report
     */
    public function generatePerformanceReport(string $moduleName, int $days = 7): array
    {
        $installs = ModuleInstallLog::where('module_name', $moduleName)
            ->where('status', 'installed')
            ->where('installed_at', '>=', now()->subDays($days))
            ->get();

        if ($installs->isEmpty()) {
            return ['message' => 'No installations found'];
        }

        return [
            'module_name' => $moduleName,
            'period_days' => $days,
            'total_installations' => $installs->count(),
            'success_rate' => round(($installs->count() / ModuleInstallLog::where('module_name', $moduleName)->count()) * 100, 2),
            'average_duration_seconds' => round($installs->avg(function ($i) {
                return now()->diffInSeconds($i->installed_at);
            }), 2),
            'validation_score_average' => round($installs->avg('validation_score'), 2),
        ];
    }
}
