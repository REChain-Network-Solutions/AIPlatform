<?php

namespace App\Models\Crm;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\SoftDeletes;

class CrmActivity extends Model
{
    use SoftDeletes;

    protected $table = 'crm_activities';

    protected $fillable = [
        'user_id',
        'type',
        'subject',
        'body',
        'due_at',
        'done_at',
        'is_done',
        'crm_contact_id',
        'crm_company_id',
        'crm_deal_id',
    ];

    protected $casts = [
        'due_at'  => 'datetime',
        'done_at' => 'datetime',
        'is_done' => 'boolean',
    ];

    public static array $typeIcons = [
        'note'     => 'tabler-notes',
        'call'     => 'tabler-phone',
        'email'    => 'tabler-mail',
        'meeting'  => 'tabler-calendar',
        'task'     => 'tabler-checkbox',
        'deadline' => 'tabler-alarm',
    ];

    public function getTypeIconAttribute(): string
    {
        return self::$typeIcons[$this->type] ?? 'tabler-activity';
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function contact(): BelongsTo
    {
        return $this->belongsTo(CrmContact::class, 'crm_contact_id');
    }

    public function company(): BelongsTo
    {
        return $this->belongsTo(CrmCompany::class, 'crm_company_id');
    }

    public function deal(): BelongsTo
    {
        return $this->belongsTo(CrmDeal::class, 'crm_deal_id');
    }

    public function markDone(): void
    {
        $this->update(['is_done' => true, 'done_at' => now()]);
    }

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopePending($query)
    {
        return $query->where('is_done', false);
    }

    public function scopeOverdue($query)
    {
        return $query->where('is_done', false)->where('due_at', '<', now());
    }
}
