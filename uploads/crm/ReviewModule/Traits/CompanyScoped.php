<?php

namespace Modules\ReviewModule\Traits;

use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Schema;

trait CompanyScoped
{
    protected static function bootCompanyScoped(): void
    {
        static::creating(function ($model) {
            if (property_exists($model, 'company_id') || Schema::hasColumn($model->getTable(), 'company_id')) {
                if (empty($model->company_id) && Auth::check() && isset(Auth::user()->company_id)) {
                    $model->company_id = Auth::user()->company_id;
                }
            }
        });

        static::addGlobalScope('company_id', function (Builder $builder) {
            try {
                $model = $builder->getModel();
                if (!Auth::check() || !isset(Auth::user()->company_id)) return;
                if (Schema::hasColumn($model->getTable(), 'company_id')) {
                    $builder->where($model->getTable().'.company_id', Auth::user()->company_id);
                }
            } catch (\Throwable $e) {
                // fail-open to avoid breaking boot if schema not ready during migrations
            }
        });
    }
}
