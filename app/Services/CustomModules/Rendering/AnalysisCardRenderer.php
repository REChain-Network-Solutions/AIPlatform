<?php
namespace App\Services\CustomModules\Rendering;

use App\Services\CustomModules\DTO\ModuleAnalysisResult;

class AnalysisCardRenderer
{
    public function emptyState(): string
    {
        return '<div class="card"><div class="card-body text-muted">No analysis has been run yet.</div></div>';
    }

    public function uploadSuccessCard(string $file): string
    {
        return '<div class="card"><div class="card-body"><strong>Upload complete:</strong> ' . e($file) . '</div></div>';
    }

    public function render(ModuleAnalysisResult $analysis): string
    {
        $rows = '';
        foreach ($analysis->checks as $key => $value) {
            $rows .= '<li><strong>' . e($key) . ':</strong> ' . e(is_bool($value) ? ($value ? 'true' : 'false') : (string) $value) . '</li>';
        }

        return '<div class="card"><div class="card-header">Analyzer summary</div><div class="card-body"><ul>' . $rows . '</ul></div></div>';
    }

    public function renderSafeFixPlan(array $result): string
    {
        $items = '';
        foreach (($result['planned_actions'] ?? []) as $item) {
            $items .= '<li>' . e($item) . '</li>';
        }

        return '<div class="card"><div class="card-header">Safe fix plan</div><div class="card-body"><ul>' . $items . '</ul></div></div>';
    }
}
