<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class RouteReferenceExtractorService
{
    public function extract(array $files, string $modulePath = ''): array
    {
        $routeNames = [];
        $routeCalls = [];
        foreach ($files as $file) {
            if (!File::exists($file)) {
                continue;
            }
            $content = File::get($file);
            preg_match_all("#route\([\'\"]([^\'\"]+)[\'\"]#", $content, $matches);
            foreach ($matches[1] ?? [] as $name) {
                $routeNames[] = $name;
                $routeCalls[] = [
                    'route' => $name,
                    'file' => $modulePath ? str_replace(rtrim($modulePath, '/') . '/', '', $file) : $file,
                ];
            }
        }

        return [
            'route_names' => array_values(array_unique($routeNames)),
            'route_calls' => $routeCalls,
        ];
    }
}
