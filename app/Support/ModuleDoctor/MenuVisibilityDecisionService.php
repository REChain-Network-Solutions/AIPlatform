<?php

namespace App\Support\ModuleDoctor;

class MenuVisibilityDecisionService
{
    public function decide(array $audit): array
    {
        if (!($audit['provider']['ready'] ?? false)) {
            return ['code' => 'PROVIDER_BOOT_FAILURE', 'visible' => false];
        }
        if (!($audit['routes']['ready'] ?? false)) {
            return ['code' => 'BAD_ROUTE', 'visible' => false];
        }
        if (!($audit['sidebar']['has_sidebar'] ?? false) && ($audit['routes']['ready'] ?? false)) {
            return ['code' => 'SIDEBAR_MISSING_BUT_ROUTES_VALID', 'visible' => false];
        }
        if (!($audit['menu']['ready'] ?? false)) {
            return ['code' => 'MENU_REGISTRATION_MISSING', 'visible' => false];
        }
        if (!($audit['permissions']['ready'] ?? false)) {
            return ['code' => 'HIDDEN_BY_PERMISSION', 'visible' => false];
        }
        if (!($audit['package']['ready'] ?? false)) {
            return ['code' => 'TENANT_BLOCKED', 'visible' => false];
        }

        return ['code' => 'VISIBLE', 'visible' => true];
    }
}
