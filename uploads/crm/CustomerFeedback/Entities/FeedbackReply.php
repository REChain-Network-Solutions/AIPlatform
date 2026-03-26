<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Scopes\ActiveScope;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\SoftDeletes;

/**
 * FeedbackReply - Thread responses to feedback tickets
 * 
 * @property int $id
 * @property int $feedback_id (parent FeedbackTicket)
 * @property int $user_id (responder)
 * @property string $message
 * @property string|null $message_html (enriched format)
 * @property boolean $is_internal (visible to agents only)
 * @property boolean $is_ai_generated (auto-response flag)
 * @property string|null $email_message_id (for email threading)
 * @property string|null $source_channel (email|portal|api|auto)
 * @property json|null $metadata
 * @property datetime $created_at
 * @property datetime $updated_at
 * @property datetime|null $deleted_at
 */
class FeedbackReply extends BaseModel
{
    use HasCompany;
    use SoftDeletes;

    protected $table = 'feedback_replies';
    protected $dates = ['deleted_at'];
    protected $casts = [
        'is_internal' => 'boolean',
        'is_ai_generated' => 'boolean',
        'metadata' => 'array',
    ];

    const SOURCE_EMAIL = 'email';
    const SOURCE_PORTAL = 'portal';
    const SOURCE_API = 'api';
    const SOURCE_AUTO = 'auto';

    /**
     * Get the parent ticket
     */
    public function ticket(): BelongsTo
    {
        return $this->belongsTo(FeedbackTicket::class, 'feedback_id');
    }

    /**
     * Get the user who wrote this reply
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id')->withoutGlobalScope(ActiveScope::class);
    }

    /**
     * Get attached files to this reply
     */
    public function files(): HasMany
    {
        return $this->hasMany(FeedbackFile::class, 'reply_id');
    }

    /**
     * Scope: Get only visible replies (not internal)
     */
    public function scopePublic($query)
    {
        return $query->where('is_internal', false);
    }

    /**
     * Scope: Get only internal replies
     */
    public function scopeInternal($query)
    {
        return $query->where('is_internal', true);
    }

    /**
     * Scope: Get only AI-generated replies
     */
    public function scopeAiGenerated($query)
    {
        return $query->where('is_ai_generated', true);
    }
}
