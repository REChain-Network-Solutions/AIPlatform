<?php
namespace App\Support\ModuleDoctor;

class ModuleDoctorCompatPatcher
{
    public static function map(): array
    {
        return [
            'VisibilityAuditService::build()' => 'compat-ready',
            'RegisteredRouteCatalogService::all()' => 'compat-ready',
            'upload_listing_refresh' => 'enabled',
            'doctor_summary_cards' => 'enabled',
            'analysis_card_renderer' => 'enabled',
        ];
    }
}
