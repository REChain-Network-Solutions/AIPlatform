<?php
namespace App\Support\ModuleDoctor;
class InstallBlockerRegistryService {
    public function blockers(array $audit): array {
        $blockers = [];
        foreach (['provider','routes','menu','permissions','package'] as $key) {
            if (!(bool)($audit[$key]['ready'] ?? false)) {
                $blockers[] = $key;
            }
        }
        return $blockers;
    }
}
