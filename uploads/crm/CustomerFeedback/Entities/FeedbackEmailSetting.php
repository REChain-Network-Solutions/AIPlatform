<?php

namespace Modules\CustomerFeedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

class FeedbackEmailSetting extends BaseModel
{
    use HasCompany;

    protected $table = 'feedback_email_settings';
    protected $fillable = [
        'imap_host',
        'imap_port',
        'imap_encryption',
        'imap_username',
        'imap_password',
        'email_address',
        'auto_reply',
        'reply_message',
        'last_sync',
        'company_id'
    ];

    protected $casts = [
        'auto_reply' => 'boolean',
        'last_sync' => 'datetime',
    ];

    protected $hidden = ['imap_password'];
}
