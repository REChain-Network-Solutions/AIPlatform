<?php

namespace App\Models\Crm;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\MorphMany;
use Illuminate\Database\Eloquent\SoftDeletes;

class CrmContact extends Model
{
    use SoftDeletes;

    protected $table = 'crm_contacts';

    protected $fillable = [
        'user_id',
        'first_name',
        'last_name',
        'email',
        'phone',
        'job_title',
        'company_name',
        'crm_company_id',
        'linkedin_url',
        'twitter_handle',
        'website',
        'address',
        'city',
        'state',
        'country',
        'avatar_path',
        'status',
        'notes',
        'custom_fields',
        'last_contacted_at',
    ];

    protected $casts = [
        'custom_fields'      => 'array',
        'last_contacted_at'  => 'datetime',
    ];

    public function getFullNameAttribute(): string
    {
        return trim("{$this->first_name} {$this->last_name}");
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function company(): BelongsTo
    {
        return $this->belongsTo(CrmCompany::class, 'crm_company_id');
    }

    public function deals(): HasMany
    {
        return $this->hasMany(CrmDeal::class, 'crm_contact_id');
    }

    public function activities(): HasMany
    {
        return $this->hasMany(CrmActivity::class, 'crm_contact_id')->latest();
    }

    public function tags(): MorphMany
    {
        return $this->morphMany(CrmTaggable::class, 'taggable');
    }

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId);
    }
}
