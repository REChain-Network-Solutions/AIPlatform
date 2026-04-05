<?php
namespace App\Services\CustomModules\Analyzers;

class ManifestAnalyzer
{
    public function inspect(string $filePath): array
    {
        return ['status' => 'ok', 'module_root' => pathinfo($filePath, PATHINFO_FILENAME), 'warnings' => [], 'blocking_issues' => []];
    }
}
