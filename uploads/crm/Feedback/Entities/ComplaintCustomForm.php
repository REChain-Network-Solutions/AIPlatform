<?php

namespace Modules\Feedback\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Models\CustomField;
use App\Traits\CustomFieldsTrait;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\FeedbackCustomForm
 *
 * @property int $id
 * @property string $field_display_name
 * @property string $field_name
 * @property string $field_type
 * @property int $field_order
 * @property string $status
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm query()
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereFieldDisplayName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereFieldName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereFieldOrder($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereFieldType($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereStatus($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int $required
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereRequired($value)
 * @property int|null $company_id
 * @property int|null $custom_fields_id
 * @property-read \App\Models\Company|null $company
 * @property-read \App\Models\CustomField|null $customField
 * @property-read mixed $extras
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereCompanyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|FeedbackCustomForm whereCustomFieldsId($value)
 */
class FeedbackCustomForm extends BaseModel
{

    use CustomFieldsTrait;
    use HasCompany;

    protected $guarded = ['id'];

    public function customField(): BelongsTo
    {
        return $this->belongsTo(CustomField::class, 'custom_fields_id');
    }

}
