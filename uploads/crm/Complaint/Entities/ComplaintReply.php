<?php

namespace Modules\Complaint\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Scopes\ActiveScope;
use Modules\Complaint\Entities\Complaint;
use Modules\Complaint\Entities\ComplaintFile;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\ComplaintReply
 *
 * @property int $id
 * @property int $complaint_id
 * @property int $user_id
 * @property string|null $message
 * @property int|null $added_by
 * @property int|null $agent_id
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property \Illuminate\Support\Carbon|null $deleted_at
 * @property-read \Illuminate\Database\Eloquent\Collection|\App\Models\ComplaintFile[] $files
 * @property-read int|null $files_count
 * @property-read mixed $icon
 * @property-read \App\Models\Complaint $complaint
 * @property-read \App\Models\User $user
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply newQuery()
 * @method static \Illuminate\Database\Query\Builder|ComplaintReply onlyTrashed()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereDeletedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereMessage($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereComplaintId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereUpdatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereUserId($value)
 * @method static \Illuminate\Database\Query\Builder|ComplaintReply withTrashed()
 * @method static \Illuminate\Database\Query\Builder|ComplaintReply withoutTrashed()
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property string|null $imap_message_id
 * @property string|null $imap_message_uid
 * @property string|null $imap_in_reply_to
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereImapInReplyTo($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereImapMessageId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReply whereImapMessageUid($value)
 */
class ComplaintReply extends BaseModel
{

    use SoftDeletes;

    protected $dates = ['deleted_at'];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function files(): HasMany
    {
        return $this->hasMany(ComplaintFile::class, 'complaint_reply_id');
    }

    public function complaint(): BelongsTo
    {
        return $this->belongsTo(Complaint::class);
    }

}
