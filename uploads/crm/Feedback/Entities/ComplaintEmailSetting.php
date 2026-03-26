<?php

namespace Modules\Feedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

/**
 * App\Models\FeedbackEmailSetting
 *
 * @property int $id
 * @property int|null $company_id
 * @property string|null $mail_username
 * @property string|null $mail_password
 * @property string|null $mail_from_name
 * @property string|null $mail_from_email
 * @property string|null $imap_host
 * @property string|null $imap_port
 * @property string|null $imap_encryption
 * @property int $status
 * @property int $verified
 * @property int $sync_interval
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereCompanyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereImapEncryption($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereImapHost($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereImapPort($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereMailFromEmail($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereMailFromName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereMailPassword($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereMailUsername($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereStatus($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereSyncInterval($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereUpdatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackEmailSetting whereVerified($value)
 * @mixin \Eloquent
 * @property-read \App\Models\Company|null $company
 */
class FeedbackEmailSetting extends BaseModel
{

    use HasCompany;

    protected $guarded = ['id'];
    protected $table = 'feedback_email_settings';

    public static function createEmailSetting($company)
    {
        FeedbackEmailSetting::create([
            'company_id' => $company->id,
            'status' => 0,
            'verified' => 0,
        ]);
    }
}
