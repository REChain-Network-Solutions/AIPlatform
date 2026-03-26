<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Scopes\ActiveScope;
use App\Models\ModuleSetting;
use App\Traits\CustomFieldsTrait;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;

/**
 * FeedbackTicket - Unified entity for complaints, feedback, and survey responses
 * 
 * @property int $id
 * @property int $company_id
 * @property int $user_id (requester/client)
 * @property int|null $agent_id (assigned agent)
 * @property string $title
 * @property string $description
 * @property enum $status (open|in_progress|resolved|closed|pending)
 * @property enum $priority (low|medium|high|critical)
 * @property enum $feedback_type (complaint|feedback|survey_response)
 * @property int|null $channel_id (email|web_form|api|sms|chat)
 * @property int|null $group_id
 * @property int|null $type_id (feedback type classification)
 * @property int|null $nps_score (1-10 for NPS surveys)
 * @property int|null $csat_score (1-5 for CSAT surveys)
 * @property json|null $custom_meta (flexible field storage)
 * @property json|null $ai_metadata (AI analysis results)
 * @property string|null $email_thread_id (for email sync)
 * @property boolean $read (agent read status)
 * @property datetime|null $resolved_at
 * @property datetime|null $deleted_at
 * @property datetime $created_at
 * @property datetime $updated_at
 */
class FeedbackTicket extends BaseModel
{
    use HasCompany;
    use SoftDeletes, HasFactory;
    use CustomFieldsTrait;

    protected $table = 'feedback_tickets';
    protected $dates = ['deleted_at', 'resolved_at'];
    protected $appends = ['created_on'];
    protected $casts = [
        'custom_meta' => 'array',
        'ai_metadata' => 'array',
        'read' => 'boolean',
    ];

    const MODULE_NAME = 'customer-feedback';
    const CUSTOM_FIELD_MODEL = 'Modules\CustomerFeedback\Entities\FeedbackTicket';

    // Feedback type constants
    const TYPE_COMPLAINT = 'complaint';
    const TYPE_FEEDBACK = 'feedback';
    const TYPE_SURVEY_RESPONSE = 'survey_response';

    // Status constants
    const STATUS_OPEN = 'open';
    const STATUS_IN_PROGRESS = 'in_progress';
    const STATUS_RESOLVED = 'resolved';
    const STATUS_CLOSED = 'closed';
    const STATUS_PENDING = 'pending';

    // Priority constants
    const PRIORITY_LOW = 'low';
    const PRIORITY_MEDIUM = 'medium';
    const PRIORITY_HIGH = 'high';
    const PRIORITY_CRITICAL = 'critical';

    /**
     * Add module settings for roles
     */
    public static function addModuleSetting($company)
    {
        $roles = ['client', 'employee', 'admin'];
        ModuleSetting::createRoleSettingEntry('customer-feedback', $roles, $company);
    }

    /**
     * Get the user who created this ticket (requester/client)
     */
    public function requester(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id')->withoutGlobalScope(ActiveScope::class);
    }

    /**
     * Get the assigned agent
     */
    public function agent(): BelongsTo
    {
        return $this->belongsTo(User::class, 'agent_id')->withoutGlobalScope(ActiveScope::class);
    }

    /**
     * Alias for requester (legacy support)
     */
    public function client(): BelongsTo
    {
        return $this->requester();
    }

    /**
     * Get all replies to this ticket
     */
    public function replies(): HasMany
    {
        return $this->hasMany(FeedbackReply::class, 'feedback_id');
    }

    /**
     * Get direct tags on this ticket
     */
    public function tags(): HasMany
    {
        return $this->hasMany(FeedbackTag::class, 'feedback_id');
    }

    /**
     * Get many-to-many tag relationships
     */
    public function feedbackTags(): BelongsToMany
    {
        return $this->belongsToMany(FeedbackTagList::class, 'feedback_tags', 'feedback_id', 'tag_id');
    }

    /**
     * Get attached files
     */
    public function files(): HasMany
    {
        return $this->hasMany(FeedbackFile::class, 'feedback_id');
    }

    /**
     * Get the channel this ticket came from
     */
    public function channel(): BelongsTo
    {
        return $this->belongsTo(FeedbackChannel::class, 'channel_id');
    }

    /**
     * Get the ticket type/category
     */
    public function ticketType(): BelongsTo
    {
        return $this->belongsTo(FeedbackType::class, 'type_id');
    }

    /**
     * Get the assigned group
     */
    public function group(): BelongsTo
    {
        return $this->belongsTo(FeedbackGroup::class, 'group_id');
    }

    /**
     * Get NPS response if this is an NPS survey response
     */
    public function npsResponse(): BelongsTo
    {
        return $this->belongsTo(NpsResponse::class, 'id', 'feedback_ticket_id');
    }

    /**
     * Get CSAT response if this is a CSAT survey response
     */
    public function csatResponse(): BelongsTo
    {
        return $this->belongsTo(CsatResponse::class, 'id', 'feedback_ticket_id');
    }

    /**
     * Get AI insights for this ticket
     */
    public function insights(): HasMany
    {
        return $this->hasMany(FeedbackInsight::class, 'feedback_ticket_id');
    }

    /**
     * Check if this is a complaint
     */
    public function isComplaint(): bool
    {
        return $this->feedback_type === self::TYPE_COMPLAINT;
    }

    /**
     * Check if this is general feedback
     */
    public function isFeedback(): bool
    {
        return $this->feedback_type === self::TYPE_FEEDBACK;
    }

    /**
     * Check if this is a survey response
     */
    public function isSurveyResponse(): bool
    {
        return $this->feedback_type === self::TYPE_SURVEY_RESPONSE;
    }

    /**
     * Check if ticket is resolved
     */
    public function isResolved(): bool
    {
        return $this->status === self::STATUS_RESOLVED || $this->status === self::STATUS_CLOSED;
    }

    /**
     * Get formatted creation date
     */
    public function getCreatedOnAttribute()
    {
        $setting = company();

        if (!is_null($this->created_at)) {
            return $this->created_at->timezone($setting->timezone)->format('d M Y H:i');
        }

        return '';
    }

    /**
     * Scope: Get only complaints
     */
    public function scopeComplaints($query)
    {
        return $query->where('feedback_type', self::TYPE_COMPLAINT);
    }

    /**
     * Scope: Get only feedback
     */
    public function scopeFeedback($query)
    {
        return $query->where('feedback_type', self::TYPE_FEEDBACK);
    }

    /**
     * Scope: Get only survey responses
     */
    public function scopeSurveyResponses($query)
    {
        return $query->where('feedback_type', self::TYPE_SURVEY_RESPONSE);
    }

    /**
     * Scope: Get only unresolved tickets
     */
    public function scopeUnresolved($query)
    {
        return $query->whereNotIn('status', [self::STATUS_RESOLVED, self::STATUS_CLOSED]);
    }

    /**
     * Scope: Get high priority tickets
     */
    public function scopeHighPriority($query)
    {
        return $query->whereIn('priority', [self::PRIORITY_HIGH, self::PRIORITY_CRITICAL]);
    }
}
