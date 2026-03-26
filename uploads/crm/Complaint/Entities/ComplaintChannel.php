<?php

namespace Modules\Complaint\Entities;

use Modules\Complaint\Entities\Complaint;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * App\Models\ComplaintChannel
 *
 * @property int $id
 * @property string $channel_name
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @property-read \Illuminate\Database\Eloquent\Collection|\App\Models\Complaint[] $complaint
 * @property-read int|null $complaint_count
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel whereChannelName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintChannel whereCompanyId($value)
 */
class ComplaintChannel extends BaseModel
{

    use HasCompany;

    public function complaint(): HasMany
    {
        return $this->hasMany(Complaint::class, 'channel_id');
    }

}
