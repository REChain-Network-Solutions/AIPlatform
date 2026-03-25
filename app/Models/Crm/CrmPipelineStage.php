<?php

namespace App\Models\Crm;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class CrmPipelineStage extends Model
{
    protected $table = 'crm_pipeline_stages';

    protected $fillable = [
        'pipeline_id',
        'name',
        'color',
        'probability',
        'order',
        'is_won',
        'is_lost',
    ];

    protected $casts = [
        'is_won'  => 'boolean',
        'is_lost' => 'boolean',
    ];

    public function pipeline(): BelongsTo
    {
        return $this->belongsTo(CrmPipeline::class, 'pipeline_id');
    }

    public function deals(): HasMany
    {
        return $this->hasMany(CrmDeal::class, 'stage_id');
    }
}
