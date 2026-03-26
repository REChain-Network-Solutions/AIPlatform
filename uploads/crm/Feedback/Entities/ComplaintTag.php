<?php

namespace Modules\Feedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Modules\Feedback\Entities\FeedbackTagList;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\FeedbackTag
 *
 * @property int $id
 * @property int $tag_id
 * @property int $feedback_id
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @property-read \App\Models\FeedbackTagList $tag
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag whereTagId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag whereFeedbackId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackTag whereCompanyId($value)
 */
class FeedbackTag extends BaseModel
{

    use HasCompany;

    protected $guarded = ['id'];

    public function tag(): BelongsTo
    {
        return $this->belongsTo(FeedbackTagList::class, 'tag_id');
    }

}
