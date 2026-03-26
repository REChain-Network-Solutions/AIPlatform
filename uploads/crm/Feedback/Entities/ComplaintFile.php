<?php

namespace Modules\Feedback\Entities;

use App\Models\BaseModel;
use App\Traits\IconTrait;
use Modules\Feedback\Entities\FeedbackReply;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\FeedbackFile
 *
 * @property int $id
 * @property int $user_id
 * @property int $feedback_reply_id
 * @property string $filename
 * @property string|null $description
 * @property string|null $google_url
 * @property string|null $hashname
 * @property string|null $size
 * @property string|null $dropbox_link
 * @property string|null $external_link
 * @property string|null $external_link_name
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $file_url
 * @property-read mixed $icon
 * @property-read \App\Models\FeedbackReply $reply
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereDescription($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereDropboxLink($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereExternalLink($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereExternalLinkName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereFilename($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereGoogleUrl($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereHashname($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereSize($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereFeedbackReplyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereUpdatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackFile whereUserId($value)
 * @mixin \Eloquent
 */
class FeedbackFile extends BaseModel
{

    use IconTrait;

    const FILE_PATH = 'feedback-files';

    protected $appends = ['file_url', 'icon'];

    public function getFileUrlAttribute()
    {
        return (!is_null($this->external_link)) ? $this->external_link : asset_url_local_s3('feedback-files/' . $this->feedback_reply_id . '/' . $this->hashname);
    }

    public function reply(): BelongsTo
    {
        return $this->belongsTo(FeedbackReply::class);
    }

}
