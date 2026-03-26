<?php

namespace Modules\Feedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use Modules\Feedback\Entities\FeedbackGroup;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\FeedbackAgentGroups
 *
 * @property int $id
 * @property int $agent_id
 * @property int|null $group_id
 * @property string $status
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @property-read \App\Models\FeedbackGroup|null $group
 * @property-read \App\Models\User $user
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereAgentId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereGroupId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereStatus($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property int|null $added_by
 * @property int|null $last_updated_by
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereAddedBy($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereCompanyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackAgentGroups whereLastUpdatedBy($value)
 */
class FeedbackAgentGroups extends BaseModel
{

    use HasCompany;

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'agent_id');
    }

    public function group(): BelongsTo
    {
        return $this->belongsTo(FeedbackGroup::class, 'group_id');
    }

}
