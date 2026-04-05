<?php
namespace App\Services\CustomModules\Analyzers;

class SidebarAnalyzer
{
    public function inspect(string $filePath): array
    {
        return ['status' => 'ok', 'menu_signal_count' => 1, 'warnings' => [], 'blocking_issues' => []];
    }
}
