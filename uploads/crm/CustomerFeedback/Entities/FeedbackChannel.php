<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * FeedbackChannel - Communication channels (email, web form, API, SMS, etc.)
 */
class FeedbackChannel extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_channels';
    protected $fillable = ['name', 'slug', 'description', 'icon', 'status', 'company_id'];
    protected $casts = [
        'status' => 'boolean',
    ];

    const CHANNEL_EMAIL = 'email';
    const CHANNEL_WEB_FORM = 'web_form';
    const CHANNEL_API = 'api';
    const CHANNEL_SMS = 'sms';
    const CHANNEL_CHAT = 'chat';
    const CHANNEL_PHONE = 'phone';

    public function tickets(): HasMany
    {
        return $this->hasMany(FeedbackTicket::class, 'channel_id');
    }
}
