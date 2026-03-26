<?php

namespace Modules\Complaint\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use Modules\Complaint\Entities\ComplaintGroup;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\ComplaintAgentGroups
 *
 * @property int $id
 * @property int $agent_id
 * @property int|null $group_id
 * @property string $status
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @property-read \App\Models\ComplaintGroup|null $group
 * @property-read \App\Models\User $user
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereAgentId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereGroupId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereStatus($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property int|null $added_by
 * @property int|null $last_updated_by
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereAddedBy($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereCompanyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintAgentGroups whereLastUpdatedBy($value)
 */
class ComplaintAgentGroups extends BaseModel
{

    use HasCompany;

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'agent_id');
    }

    public function group(): BelongsTo
    {
        return $this->belongsTo(ComplaintGroup::class, 'group_id');
    }

}
