<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Modules\Complaint\Entities\ComplaintTagList;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\ComplaintTag
 *
 * @property int $id
 * @property int $tag_id
 * @property int $complaint_id
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @property-read \App\Models\ComplaintTagList $tag
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag whereTagId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag whereComplaintId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTag whereCompanyId($value)
 */
class ComplaintTag extends BaseModel
{

    use HasCompany;

    protected $guarded = ['id'];

    public function tag(): BelongsTo
    {
        return $this->belongsTo(ComplaintTagList::class, 'tag_id');
    }

}
