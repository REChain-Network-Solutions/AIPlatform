<?php
namespace App\Support\ModuleDoctor;
class MenuDiffPreviewService {
    public function diff(array $expected, array $actual): array {
        return ['missing' => array_values(array_diff($expected, $actual)), 'extra' => array_values(array_diff($actual, $expected))];
    }
}
