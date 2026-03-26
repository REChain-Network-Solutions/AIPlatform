<?php

namespace Modules\Feedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Scopes\ActiveScope;
use Modules\Feedback\Entities\Feedback;
use Modules\Feedback\Entities\FeedbackFile;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\FeedbackReply
 *
 * @property int $id
 * @property int $feedback_id
 * @property int $user_id
 * @property string|null $message
 * @property int|null $added_by
 * @property int|null $agent_id
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property \Illuminate\Support\Carbon|null $deleted_at
 * @property-read \Illuminate\Database\Eloquent\Collection|\App\Models\FeedbackFile[] $files
 * @property-read int|null $files_count
 * @property-read mixed $icon
 * @property-read \App\Models\Feedback $feedback
 * @property-read \App\Models\User $user
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply newQuery()
 * @method static \Illuminate\Database\Query\Builder|FeedbackReply onlyTrashed()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereDeletedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereMessage($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereFeedbackId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereUpdatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereUserId($value)
 * @method static \Illuminate\Database\Query\Builder|FeedbackReply withTrashed()
 * @method static \Illuminate\Database\Query\Builder|FeedbackReply withoutTrashed()
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property string|null $imap_message_id
 * @property string|null $imap_message_uid
 * @property string|null $imap_in_reply_to
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereImapInReplyTo($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereImapMessageId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackReply whereImapMessageUid($value)
 */
class FeedbackReply extends BaseModel
{

    use SoftDeletes;

    protected $dates = ['deleted_at'];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function files(): HasMany
    {
        return $this->hasMany(FeedbackFile::class, 'feedback_reply_id');
    }

    public function feedback(): BelongsTo
    {
        return $this->belongsTo(Feedback::class);
    }

}
