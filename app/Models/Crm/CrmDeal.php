<?php

namespace App\Models\Crm;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\MorphMany;
use Illuminate\Database\Eloquent\SoftDeletes;

class CrmDeal extends Model
{
    use SoftDeletes;

    protected $table = 'crm_deals';

    protected $fillable = [
        'user_id',
        'pipeline_id',
        'stage_id',
        'crm_contact_id',
        'crm_company_id',
        'title',
        'value',
        'currency',
        'probability',
        'expected_close_date',
        'closed_at',
        'status',
        'description',
        'lost_reason',
        'order',
        'custom_fields',
    ];

    protected $casts = [
        'value'               => 'decimal:2',
        'expected_close_date' => 'date',
        'closed_at'           => 'date',
        'custom_fields'       => 'array',
    ];

    public function getWeightedValueAttribute(): float
    {
        $prob = $this->probability ?? $this->stage?->probability ?? 0;

        return round((float) $this->value * ($prob / 100), 2);
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function pipeline(): BelongsTo
    {
        return $this->belongsTo(CrmPipeline::class, 'pipeline_id');
    }

    public function stage(): BelongsTo
    {
        return $this->belongsTo(CrmPipelineStage::class, 'stage_id');
    }

    public function contact(): BelongsTo
    {
        return $this->belongsTo(CrmContact::class, 'crm_contact_id');
    }

    public function company(): BelongsTo
    {
        return $this->belongsTo(CrmCompany::class, 'crm_company_id');
    }

    public function activities(): HasMany
    {
        return $this->hasMany(CrmActivity::class, 'crm_deal_id')->latest();
    }

    public function tags(): MorphMany
    {
        return $this->morphMany(CrmTaggable::class, 'taggable');
    }

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeOpen($query)
    {
        return $query->where('status', 'open');
    }
}
