<?php

namespace App\Services\CustomModules;

class TemplateRegistry
{
    /**
     * Required templates for installation UI
     */
    public const REQUIRED_TEMPLATES = [
        'install.blade.php' => 'Main upload form and status display',
        'upload-list.blade.php' => 'List of uploaded module files',
        'results-panel.blade.php' => 'Installation result and diagnostics',
    ];
    
    /**
     * Optional diagnostic templates
     */
    public const DIAGNOSTIC_TEMPLATES = [
        'menu-diff.blade.php' => 'Preview of menu changes',
        'permission-matrix.blade.php' => 'Permission mapping view',
        'route-truth.blade.php' => 'Route registration proof',
        'sidebar-readiness.blade.php' => 'Sidebar injection readiness',
        'doctor-panels.blade.php' => 'Doctor diagnostic panels',
    ];
    
    /**
     * Required data for each template
     */
    public const REQUIRED_TEMPLATE_DATA = [
        'install.blade.php' => [
            'moduleFiles',
            'uploadMaxFilesize',
            'doctorSummaryCards',
            'pageTitle',
        ],
        'upload-list.blade.php' => [
            'moduleFiles',
        ],
        'results-panel.blade.php' => [
            'analysis',
            'status',
            'message',
        ],
        'menu-diff.blade.php' => [
            'menu_diff',
            'existing_menus',
        ],
        'permission-matrix.blade.php' => [
            'permissions',
            'conflicts',
        ],
        'route-truth.blade.php' => [
            'routes',
            'registered_routes',
        ],
    ];
    
    /**
     * Validate that a template has all required data
     * 
     * @throws \Exception if required data is missing
     */
    public static function validate(string $templateName, array $data): void
    {
        if (!isset(self::REQUIRED_TEMPLATE_DATA[$templateName])) {
            // Template has no required data, skip validation
            return;
        }
        
        $required = self::REQUIRED_TEMPLATE_DATA[$templateName];
        $provided = array_keys($data);
        $missing = array_diff($required, $provided);
        
        if (!empty($missing)) {
            throw new \Exception(
                "Template '{$templateName}' is missing required data: " . 
                implode(', ', $missing) . 
                ". Provided: " . implode(', ', $provided)
            );
        }
    }
    
    /**
     * Get all required templates
     */
    public static function requiredTemplates(): array
    {
        return self::REQUIRED_TEMPLATES;
    }
    
    /**
     * Get all optional diagnostic templates
     */
    public static function diagnosticTemplates(): array
    {
        return self::DIAGNOSTIC_TEMPLATES;
    }
    
    /**
     * Check if a template is required
     */
    public static function isRequired(string $templateName): bool
    {
        return isset(self::REQUIRED_TEMPLATES[$templateName]);
    }
    
    /**
     * Check if a template is optional diagnostic
     */
    public static function isDiagnostic(string $templateName): bool
    {
        return isset(self::DIAGNOSTIC_TEMPLATES[$templateName]);
    }
    
    /**
     * Get required data for a template
     */
    public static function getRequiredData(string $templateName): array
    {
        return self::REQUIRED_TEMPLATE_DATA[$templateName] ?? [];
    }
    
    /**
     * Get description for a template
     */
    public static function getDescription(string $templateName): string
    {
        return self::REQUIRED_TEMPLATES[$templateName] ?? 
               self::DIAGNOSTIC_TEMPLATES[$templateName] ?? 
               'Unknown template';
    }
}
