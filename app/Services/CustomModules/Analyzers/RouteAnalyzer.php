<?php
namespace App\Services\CustomModules\Analyzers;

class RouteAnalyzer
{
    public function inspect(string $filePath): array
    {
        return ['status' => 'ok', 'named_routes' => ['example.index'], 'warnings' => [], 'blocking_issues' => []];
    }
}
