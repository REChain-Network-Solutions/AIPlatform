<?php

namespace App\Models\Crm;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\MorphMany;
use Illuminate\Database\Eloquent\SoftDeletes;

class CrmLead extends Model
{
    use SoftDeletes;

    protected $table = 'crm_leads';

    protected $fillable = [
        'user_id',
        'crm_lead_status_id',
        'crm_lead_source_id',
        'title',
        'contact_name',
        'contact_email',
        'contact_phone',
        'company_name',
        'website',
        'value',
        'currency',
        'description',
        'custom_fields',
        'order',
        'assigned_to_user_id',
        'converted_to_contact_id',
        'converted_to_deal_id',
        'converted_at',
    ];

    protected $casts = [
        'value'        => 'decimal:2',
        'custom_fields' => 'array',
        'converted_at' => 'datetime',
    ];

    public function getIsConvertedAttribute(): bool
    {
        return $this->converted_at !== null;
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function status(): BelongsTo
    {
        return $this->belongsTo(CrmLeadStatus::class, 'crm_lead_status_id');
    }

    public function source(): BelongsTo
    {
        return $this->belongsTo(CrmLeadSource::class, 'crm_lead_source_id');
    }

    public function assignedTo(): BelongsTo
    {
        return $this->belongsTo(User::class, 'assigned_to_user_id');
    }

    public function convertedContact(): BelongsTo
    {
        return $this->belongsTo(CrmContact::class, 'converted_to_contact_id');
    }

    public function convertedDeal(): BelongsTo
    {
        return $this->belongsTo(CrmDeal::class, 'converted_to_deal_id');
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
        return $query->whereNull('converted_at');
    }
}
