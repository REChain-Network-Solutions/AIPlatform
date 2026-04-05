<?php
namespace App\Support\ModuleDoctor;
class RouteIndexPolicyService {
    public function validate(array $routeNames): array {
        $bad = array_values(array_filter($routeNames, fn ($name) => !str_ends_with($name, '.index') && !str_contains($name, '.dashboard')));
        return ['bad' => $bad, 'ready' => empty($bad)];
    }
}
