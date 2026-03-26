<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

/**
 * App\Models\ComplaintEmailSetting
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
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereCompanyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereImapEncryption($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereImapHost($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereImapPort($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereMailFromEmail($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereMailFromName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereMailPassword($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereMailUsername($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereStatus($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereSyncInterval($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereUpdatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintEmailSetting whereVerified($value)
 * @mixin \Eloquent
 * @property-read \App\Models\Company|null $company
 */
class ComplaintEmailSetting extends BaseModel
{

    use HasCompany;

    protected $guarded = ['id'];
    protected $table = 'complaint_email_settings';

    public static function createEmailSetting($company)
    {
        ComplaintEmailSetting::create([
            'company_id' => $company->id,
            'status' => 0,
            'verified' => 0,
        ]);
    }
}
