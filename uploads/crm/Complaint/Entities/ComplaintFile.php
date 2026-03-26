<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\IconTrait;
use Modules\Complaint\Entities\ComplaintReply;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\ComplaintFile
 *
 * @property int $id
 * @property int $user_id
 * @property int $complaint_reply_id
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
 * @property-read \App\Models\ComplaintReply $reply
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereDescription($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereDropboxLink($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereExternalLink($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereExternalLinkName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereFilename($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereGoogleUrl($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereHashname($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereSize($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereComplaintReplyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereUpdatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintFile whereUserId($value)
 * @mixin \Eloquent
 */
class ComplaintFile extends BaseModel
{

    use IconTrait;

    const FILE_PATH = 'complaint-files';

    protected $appends = ['file_url', 'icon'];

    public function getFileUrlAttribute()
    {
        return (!is_null($this->external_link)) ? $this->external_link : asset_url_local_s3('complaint-files/' . $this->complaint_reply_id . '/' . $this->hashname);
    }

    public function reply(): BelongsTo
    {
        return $this->belongsTo(ComplaintReply::class);
    }

}
