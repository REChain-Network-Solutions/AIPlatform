<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class FeedbackAgentGroups extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_agent_groups';
    protected $fillable = ['group_id', 'agent_id', 'added_by', 'company_id'];

    public function group(): BelongsTo
    {
        return $this->belongsTo(FeedbackGroup::class, 'group_id');
    }

    public function agent(): BelongsTo
    {
        return $this->belongsTo(User::class, 'agent_id');
    }

    public function addedBy(): BelongsTo
    {
        return $this->belongsTo(User::class, 'added_by');
    }
}
