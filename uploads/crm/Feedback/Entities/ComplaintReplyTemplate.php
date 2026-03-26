<?php

namespace Modules\Feedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

/**
 * App\Models\FeedbackReplyTemplate
 *
 * @property int $id
 * @property string $reply_heading
 * @property string $reply_text
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate whereReplyHeading($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate whereReplyText($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReplyTemplate whereCompanyId($value)
 */
class FeedbackReplyTemplate extends BaseModel
{

    use HasCompany;

    protected $table = 'feedback_reply_templates';

    protected $guarded = ['id'];

    //
}
