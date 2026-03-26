<?php

namespace Modules\Feedback\Entities;

use App\Models\User;
use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Scopes\ActiveScope;
use App\Models\ModuleSetting;
use App\Traits\CustomFieldsTrait;
use Modules\Feedback\Entities\FeedbackTag;
use Modules\Feedback\Entities\FeedbackReply;
use Modules\Feedback\Entities\FeedbackTagList;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;

class Feedback extends BaseModel
{
    use HasCompany;
    use SoftDeletes, HasFactory;
    use CustomFieldsTrait;
    
    protected $dates = ['deleted_at'];
    protected $appends = ['created_on'];
    protected $table = 'feedback';

    const MODULE_NAME = 'feedback';
    const CUSTOM_FIELD_MODEL = 'Modules\Feedback\Entities\Feedback';

    public static function addModuleSetting($company)
    {
        // create admin, employee and client module settings
        $roles = ['client', 'employee', 'admin'];
        ModuleSetting::createRoleSettingEntry('feedback', $roles, $company);
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
        return $this->hasMany(FeedbackReply::class, 'feedback_id');
    }

    public function tags(): HasMany
    {
        return $this->hasMany(FeedbackTag::class, 'feedback_id');
    }

    public function feedbackTags(): BelongsToMany
    {
        return $this->belongsToMany(FeedbackTagList::class, 'feedback_tags', 'feedback_id', 'tag_id');
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
 