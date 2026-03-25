<?php

namespace App\Models\Crm;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\MorphMany;
use Illuminate\Database\Eloquent\SoftDeletes;

class CrmCompany extends Model
{
    use SoftDeletes;

    protected $table = 'crm_companies';

    protected $fillable = [
        'user_id',
        'name',
        'domain',
        'website',
        'phone',
        'industry',
        'type',
        'employee_count',
        'annual_revenue',
        'address',
        'city',
        'state',
        'country',
        'logo_path',
        'description',
        'custom_fields',
    ];

    protected $casts = [
        'custom_fields'  => 'array',
        'annual_revenue' => 'decimal:2',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function contacts(): HasMany
    {
        return $this->hasMany(CrmContact::class, 'crm_company_id');
    }

    public function deals(): HasMany
    {
        return $this->hasMany(CrmDeal::class, 'crm_company_id');
    }

    public function activities(): HasMany
    {
        return $this->hasMany(CrmActivity::class, 'crm_company_id')->latest();
    }

    public function tags(): MorphMany
    {
        return $this->morphMany(CrmTaggable::class, 'taggable');
    }

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function getTotalDealValueAttribute(): float
    {
        return (float) $this->deals()->where('status', 'open')->sum('value');
    }
}
