<?php
namespace App\Support\ModuleDoctor;
class MenuParentResolverService {
    public function resolve(string $moduleName): array {
        $map = ['settings' => 'Settings', 'dashboard' => 'Dashboard', 'work' => 'Work'];
        return ['module' => $moduleName, 'preferred_parent' => 'work', 'label' => $map['work']];
    }
}
