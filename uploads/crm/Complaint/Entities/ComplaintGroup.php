<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Modules\Complaint\Entities\ComplaintAgentGroups;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Factories\HasFactory;

/**
 * App\Models\ComplaintGroup
 *
 * @property int $id
 * @property string $group_name
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read \Illuminate\Database\Eloquent\Collection|\App\Models\ComplaintAgentGroups[] $agents
 * @property-read int|null $agents_count
 * @property-read mixed $icon
 * @property-read mixed $enabledAgents
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup whereGroupName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @property-read int|null $enabled_agents_count
 * @method static \Database\Factories\ComplaintGroupFactory factory(...$parameters)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintGroup whereCompanyId($value)
 */
class ComplaintGroup extends BaseModel
{

    use HasFactory, HasCompany;

    public function enabledAgents(): HasMany
    {
        return $this->hasMany(ComplaintAgentGroups::class, 'group_id')->where('status', '=', 'enabled');
    }

}
