<?php

namespace Modules\Feedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Modules\Feedback\Entities\FeedbackAgentGroups;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Factories\HasFactory;

/**
 * App\Models\FeedbackGroup
 *
 * @property int $id
 * @property string $group_name
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read \Illuminate\Database\Eloquent\Collection|\App\Models\FeedbackAgentGroups[] $agents
 * @property-read int|null $agents_count
 * @property-read mixed $icon
 * @property-read mixed $enabledAgents
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup whereGroupName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @property-read int|null $enabled_agents_count
 * @method static \Database\Factories\FeedbackGroupFactory factory(...$parameters)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackGroup whereCompanyId($value)
 */
class FeedbackGroup extends BaseModel
{

    use HasFactory, HasCompany;

    public function enabledAgents(): HasMany
    {
        return $this->hasMany(FeedbackAgentGroups::class, 'group_id')->where('status', '=', 'enabled');
    }

}
