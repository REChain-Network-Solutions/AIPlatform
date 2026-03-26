<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Modules\Complaint\Entities\Complaint;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * App\Models\ComplaintType
 *
 * @property int $id
 * @property string $type
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @property-read \Illuminate\Database\Eloquent\Collection|\App\Models\Complaint[] $tickets
 * @property-read int|null $tickets_count
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType whereType($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintType whereCompanyId($value)
 */
class ComplaintType extends BaseModel
{

    use HasCompany;

    public function complaint(): HasMany
    {
        return $this->hasMany(Complaint::class, 'type_id');
    }

}
