<?php

namespace Modules\AuditLog\app\Traits;

use App\Services\AddonService\AddonService;
use OwenIt\Auditing\Auditable;

trait ConditionalAuditable
{
    use Auditable;

    /**
     * Boot the conditional auditable trait.
     */
    public static function bootConditionalAuditable()
    {
        // Check if audit should be enabled for this model
        static::creating(function ($model) {
            $model->checkAuditEnabled();
        });

        static::updating(function ($model) {
            $model->checkAuditEnabled();
        });

        static::deleting(function ($model) {
            $model->checkAuditEnabled();
        });
    }

    /**
     * Check if auditing is enabled.
     */
    protected function checkAuditEnabled()
    {
        $addonService = app(AddonService::class);

        if (! $addonService->isAddonEnabled('AuditLog')) {
            $this->disableAuditing();
        }
    }

    /**
     * Determine if the model should be audited.
     */
    public function shouldAudit(): bool
    {
        $addonService = app(AddonService::class);

        if (! $addonService->isAddonEnabled('AuditLog')) {
            return false;
        }

        return parent::shouldAudit();
    }
}
