<?php
namespace App\Support\ModuleDoctor;
class ProviderBootVerificationService {
    public function inspect(array $module): array {
        $files = $module['files'] ?? [];
        $providers = array_values(array_filter($files, fn ($file) => str_contains($file, 'ServiceProvider.php')));
        $hasRouteLoader = (new \Illuminate\Support\Collection($files))->contains(fn ($file) => str_contains($file, 'loadRoutesFrom(') || str_contains($file, 'routes/'));
        return ['providers' => $providers, 'has_provider' => !empty($providers), 'has_route_loader' => $hasRouteLoader, 'ready' => !empty($providers)];
    }
}
