<?php

namespace Modules\Feedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

/**
 * App\Models\FeedbackTagList
 *
 * @property int $id
 * @property string $tag_name
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList whereTagName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTagList whereCompanyId($value)
 */
class FeedbackTagList extends BaseModel
{

    use HasCompany;

    protected $table = 'feedback_tag_list';

    protected $guarded = ['id'];

}
