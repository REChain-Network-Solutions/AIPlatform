<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Scopes\ActiveScope;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;

/**
 * FeedbackGroup - Organizational groups for handling feedback
 */
class FeedbackGroup extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_groups';
    protected $fillable = ['name', 'description', 'status', 'company_id'];
    protected $casts = [
        'status' => 'boolean',
    ];

    public function tickets(): HasMany
    {
        return $this->hasMany(FeedbackTicket::class, 'group_id');
    }

    public function agents(): BelongsToMany
    {
        return $this->belongsToMany(
            User::class,
            'feedback_agent_groups',
            'group_id',
            'agent_id'
        )->withPivot('added_by', 'created_at')->withoutGlobalScope(ActiveScope::class);
    }

    public function enabledAgents(): BelongsToMany
    {
        return $this->agents()->where('users.active', 1);
    }
}
