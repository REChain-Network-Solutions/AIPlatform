<?php

namespace Modules\Feedback\Entities;

use Modules\Feedback\Entities\Feedback;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * App\Models\FeedbackChannel
 *
 * @property int $id
 * @property string $channel_name
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @property-read \Illuminate\Database\Eloquent\Collection|\App\Models\Feedback[] $feedback
 * @property-read int|null $feedback_count
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel whereChannelName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackChannel whereCompanyId($value)
 */
class FeedbackChannel extends BaseModel
{

    use HasCompany;

    public function feedback(): HasMany
    {
        return $this->hasMany(Feedback::class, 'channel_id');
    }

}
