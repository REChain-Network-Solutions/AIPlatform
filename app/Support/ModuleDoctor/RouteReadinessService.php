<?php

namespace App\Support\ModuleDoctor;

class RouteReadinessService
{
    public function inspect(array $module): array
    {
        $extracted = app(RouteReferenceExtractorService::class)->extract($module['files'] ?? []);
        $declaredNames = $extracted['route_names'] ?? [];
        $registered = app(RegisteredRouteCatalogService::class)->all();

        $missing = [];
        foreach ($declaredNames as $name) {
            if (!in_array($name, $registered, true)) {
                $missing[] = $name;
            }
        }

        return [
            'declared' => $extracted,
            'registered_count' => count($registered),
            'missing' => $missing,
            'ready' => empty($missing),
        ];
    }
}
