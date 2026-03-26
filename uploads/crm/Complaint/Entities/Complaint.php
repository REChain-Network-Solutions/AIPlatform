<?php

namespace Modules\Complaint\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Scopes\ActiveScope;
use App\Models\ModuleSetting;
use App\Traits\CustomFieldsTrait;
use Modules\Complaint\Entities\ComplaintTag;
use Modules\Complaint\Entities\ComplaintReply;
use Modules\Complaint\Entities\ComplaintTagList;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;

class Complaint extends BaseModel
{
    use HasCompany;
    use SoftDeletes, HasFactory;
    use CustomFieldsTrait;
    
    protected $dates = ['deleted_at'];
    protected $appends = ['created_on'];
    protected $table = 'complaint';

    const MODULE_NAME = 'complaint';
    const CUSTOM_FIELD_MODEL = 'Modules\Complaint\Entities\Complaint';

    public static function addModuleSetting($company)
    {
        // create admin, employee and client module settings
        $roles = ['client', 'employee', 'admin'];
        ModuleSetting::createRoleSettingEntry('complaint', $roles, $company);
    }

    public function requester(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function agent(): BelongsTo
    {
        return $this->belongsTo(User::class, 'agent_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function client(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id')->withoutGlobalScope(ActiveScope::class);
    }

    public function reply(): HasMany
    {
        return $this->hasMany(ComplaintReply::class, 'complaint_id');
    }

    public function tags(): HasMany
    {
        return $this->hasMany(ComplaintTag::class, 'complaint_id');
    }

    public function complaintTags(): BelongsToMany
    {
        return $this->belongsToMany(ComplaintTagList::class, 'complaint_tags', 'complaint_id', 'tag_id');
    }

    public function getCreatedOnAttribute()
    {
        $setting = company();

        if (!is_null($this->created_at)) {
            return $this->created_at->timezone($setting->timezone)->format('d M Y H:i');
        }

        return '';
    }
}
 