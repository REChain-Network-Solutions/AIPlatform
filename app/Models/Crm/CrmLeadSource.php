<?php

namespace App\Models\Crm;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class CrmLeadSource extends Model
{
    protected $table = 'crm_lead_sources';

    protected $fillable = [
        'user_id',
        'name',
        'is_active',
    ];

    protected $casts = [
        'is_active' => 'boolean',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function leads(): HasMany
    {
        return $this->hasMany(CrmLead::class, 'crm_lead_source_id');
    }

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId)->where('is_active', true)->orderBy('name');
    }
}
