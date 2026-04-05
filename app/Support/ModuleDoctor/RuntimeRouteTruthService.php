<?php
namespace App\Support\ModuleDoctor;
class RuntimeRouteTruthService {
    public function check(array $declared): array {
        $registered = app(RegisteredRouteCatalogService::class)->all();
        return [
            'declared' => $declared,
            'resolved' => array_values(array_intersect($declared, $registered)),
            'missing' => array_values(array_diff($declared, $registered)),
        ];
    }
}
