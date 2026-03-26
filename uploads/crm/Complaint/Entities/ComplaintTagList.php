<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

/**
 * App\Models\ComplaintTagList
 *
 * @property int $id
 * @property string $tag_name
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList whereTagName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintTagList whereCompanyId($value)
 */
class ComplaintTagList extends BaseModel
{

    use HasCompany;

    protected $table = 'complaint_tag_list';

    protected $guarded = ['id'];

}
